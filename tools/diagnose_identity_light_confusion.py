"""Fixed-light cross-identity audit, with common-geometry and oracle interventions."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rrnet.config import load_config, model_kwargs
from rrnet.losses import luminance
from rrnet.physical_reference_data import MEADPhysicalReferencePairs, sample_reference_theta, sample_physical_theta
from rrnet.reference_relative_model import ReferenceRelativeRRNet


def mean(value, mask):
    return ((value * mask).sum((1, 2, 3), keepdim=True)
            / (value.shape[1] * mask.sum((1, 2, 3), keepdim=True).clamp_min(1)))


def normalized(value, mask):
    return value / mean(value, mask).clamp_min(1e-5)


def tile(tensor):
    x = tensor.detach().float().clamp(0, 1)
    if x.shape[0] == 1:
        x = x.expand(3, -1, -1)
    return Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round()
                           .astype('uint8')).resize((175, 175), Image.Resampling.LANCZOS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-root', default='outputs/rrnet_identity_light_audit')
    args = parser.parse_args()
    c = load_config(args.config)
    d = c['data']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = Path(args.output_root) / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    model = ReferenceRelativeRRNet(**model_kwargs(c, args.config)).to(device).eval()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state.get('model', state))
    datasets = {split: MEADPhysicalReferencePairs(
        d['root'], split, metadata_file=d.get('metadata_file', 'metadata.csv'),
        seed=20261001, variants_per_source=1, dynamic_epoch=False,
        num_lights=c['model']['num_lights'], group_size=1)
        for split in ('train', 'val')}
    selected = []
    for split, ds in datasets.items():
        people = sorted(set(ds.people))
        if split == 'train':
            people = [people[int(i)] for i in np.linspace(0, len(people)-1, min(6,len(people)))]
        for person in people:
            indices = [i for i,p in enumerate(ds.people) if p == person]
            for index in dict.fromkeys((indices[0], indices[len(indices)//2])):
                selected.append((split, index, ds[index]))
    anchor_item = datasets['train'][datasets['train'].people.index('MEAD_front_0')]
    def gpu(item):
        return {k: v.unsqueeze(0).to(device) for k,v in item.items() if isinstance(v,torch.Tensor)}
    anchor = gpu(anchor_item)
    rng = np.random.default_rng(20261001)
    rows, interventions, pictures, shape_pictures = [], [], [], []
    with torch.inference_mode():
        anchor_depth = model.depth(anchor['source_clean'])
        face = anchor['source_light_mask']
        candidates = [sample_physical_theta(rng,c['model']['num_lights']) for _ in range(12)]
        def spatial_score(theta):
            light,_ = model.illumination_from_theta(anchor_depth,theta[None].to(device))
            return float(mean((normalized(light,face)-1).square(),face).sqrt())
        candidates.sort(key=spatial_score, reverse=True)
        theta = torch.stack([sample_reference_theta(rng,c['model']['num_lights'],'dark'),
                             sample_reference_theta(rng,c['model']['num_lights'],'normal'),
                             candidates[0], candidates[1]]).to(device)
        names = ('dark','normal','source_a','source_b')
        common_true,_ = model.illumination_from_theta(anchor_depth.expand(4,-1,-1,-1),theta)
        anchor_source = model.renderer.blend_relight(
            anchor['source_clean'] * common_true[2:3],anchor['source_clean'],anchor['relight_mask']).clamp(0,0.999)
        anchor_pred = model.estimate_light(anchor_source,face)['theta']
        anchor_source_pred,_ = model.illumination_from_theta(anchor_depth,anchor_pred)
        for number,(split,index,item) in enumerate(selected):
            tensors=gpu(item)
            clean,mask,own_face=tensors['source_clean'],tensors['relight_mask'],tensors['source_light_mask']
            depth=model.depth(clean)
            true_light,_=model.illumination_from_theta(depth.expand(4,-1,-1,-1),theta)
            lit=model.renderer.blend_relight(clean * true_light,clean.expand(4,-1,-1,-1),mask.expand(4,-1,-1,-1)).clamp(0,0.999)
            predicted_theta=model.estimate_light(lit,own_face.expand(4,-1,-1,-1))['theta']
            predicted_own,_=model.illumination_from_theta(depth.expand(4,-1,-1,-1),predicted_theta)
            predicted_common,_=model.illumination_from_theta(anchor_depth.expand(4,-1,-1,-1),predicted_theta)
            for light_index,name in enumerate(names):
                gt,pred=common_true[light_index:light_index+1],predicted_common[light_index:light_index+1]
                row=dict(split=split,person=item['source_person_id'],source_index=index,
                    clean_frame=datasets[split].samples[index]['clean_frame'],light=name,
                    clean_face_luma=float(mean(luminance(clean),own_face)),
                    lit_face_luma=float(mean(luminance(lit[light_index:light_index+1]),own_face)),
                    common_true_exposure=float(mean(gt,face)),
                    common_predicted_exposure=float(mean(pred,face)),
                    exposure_relative_error=float((mean(pred,face)/mean(gt,face)-1).abs()),
                    true_common_shape_std=float(mean((normalized(gt,face)-1).square(),face).sqrt()),
                    predicted_common_shape_std=float(mean((normalized(pred,face)-1).square(),face).sqrt()),
                    common_shape_mae=float(mean((normalized(pred,face)-normalized(gt,face)).abs(),face)),
                    own_shape_mae=float(mean((normalized(predicted_own[light_index:light_index+1],own_face)-normalized(true_light[light_index:light_index+1],own_face)).abs(),own_face)))
                rows.append(row)
                if light_index >= 2 and split=='val' and index==datasets[split].people.index(item['source_person_id']):
                    shape_pictures.append((item['source_person_id'],name,
                        [clean[0],lit[light_index],
                         ((normalized(gt,face)-0.75)/0.5)[0],
                         ((normalized(pred,face)-0.75)/0.5)[0],
                         ((normalized(true_light[light_index:light_index+1],own_face)-0.75)/0.5)[0],
                         ((normalized(predicted_own[light_index:light_index+1],own_face)-0.75)/0.5)[0]]))
                if light_index < 2:
                    source_gt,source_pred=common_true[2:3],anchor_source_pred
                    target=model.renderer.blend_relight(anchor['source_clean'] * gt,anchor['source_clean'],anchor['relight_mask']).clamp(0,0.999)
                    def render(s,r):
                        gain=model.compute_transfer_gain(anchor_source,s,r)
                        return model.renderer.blend_relight(anchor_source*gain,anchor_source,anchor['relight_mask'])
                    before=render(source_pred,pred)
                    correct_exposure=render(
                        normalized(source_pred,face)*mean(source_gt,face),
                        normalized(pred,face)*mean(gt,face))
                    correct_shape=render(normalized(source_gt,face)*mean(source_pred,face),
                                         normalized(gt,face)*mean(pred,face))
                    results=dict(split=split,reference_person=item['source_person_id'],source_index=index,mode=name)
                    for label,value in (
                        ('predicted_both',before),('correct_source',render(source_gt,pred)),
                        ('correct_reference',render(source_pred,gt)),('correct_exposure',correct_exposure),
                        ('correct_shape',correct_shape),('known_both',render(source_gt,gt))):
                        results[label]=float(mean((value-target).abs(),face))
                    results['output_luma']=float(mean(luminance(before),face))
                    results['target_luma']=float(mean(luminance(target),face))
                    interventions.append(results)
                    if split=='val' and index==datasets[split].people.index(item['source_person_id']):
                        pictures.append((item['source_person_id'],name,
                            [lit[light_index],anchor_source[0],target[0],before[0],correct_exposure[0],correct_shape[0],render(source_gt,gt)[0]]))
            print(f"Audited {number+1}/{len(selected)}: {split}/{item['source_person_id']} frame-index {index}",flush=True)
        summaries=[]
        for split in ('train','val'):
            for name in names:
                subset=[r for r in rows if r['split']==split and r['light']==name]
                ex=np.array([r['common_predicted_exposure'] for r in subset])
                clean_y=np.array([r['clean_face_luma'] for r in subset])
                correlation=float(np.corrcoef(ex,clean_y)[0,1]) if ex.std()>1e-8 and clean_y.std()>1e-8 else None
                summaries.append(dict(split=split,light=name,count=len(subset),
                    true_common_exposure=subset[0]['common_true_exposure'],
                    predicted_common_exposure_min=float(ex.min()),predicted_common_exposure_max=float(ex.max()),
                    predicted_common_exposure_cv=float(ex.std()/max(ex.mean(),1e-8)),
                    mean_exposure_relative_error=float(np.mean([r['exposure_relative_error'] for r in subset])),
                    mean_common_shape_mae=float(np.mean([r['common_shape_mae'] for r in subset])),
                    true_common_shape_std=subset[0]['true_common_shape_std'],
                    mean_predicted_common_shape_std=float(np.mean([r['predicted_common_shape_std'] for r in subset])),
                    clean_luma_exposure_correlation=correlation))
        intervention_summary={split:{key:float(np.mean([r[key] for r in interventions if r['split']==split]))
            for key in ('predicted_both','correct_source','correct_reference','correct_exposure','correct_shape','known_both')}
            for split in ('train','val')}
    report=dict(checkpoint=str(Path(args.checkpoint).resolve()),config=str(Path(args.config).resolve()),
        fixed_theta={name:theta[i].cpu().tolist() for i,name in enumerate(names)},
        source_anchor_person='MEAD_front_0',anchor_source_theta=theta[2].cpu().tolist(),
        rows=rows,summary=summaries,interventions=interventions,intervention_summary=intervention_summary,
        scope='Read-only synthetic diagnostic, not final test. Same theta is a synthetic multiplier over MEAD clean lighting, not measured intrinsic illumination.',
        interpretation='Common-depth rendering removes geometry differences; correlation is descriptive and does not prove skin-tone causation. Exposure/shape corrections use ground truth and are not deployable.')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for mode in ('normal','dark'):
        entries=[entry for entry in pictures if entry[1]==mode]
        canvas=Image.new('RGB',(1240,len(entries)*205+55),'#171717')
        draw=ImageDraw.Draw(canvas)
        for column,label in enumerate(['Reference (same theta)','Fixed input','Synthetic target','Predicted','Correct exposure','Correct shape','Known theta']):
            draw.text((column*177+3,8),label,fill='white')
        draw.text((3,30),'Same input / same known reference theta; reference person changes. Oracle corrections are diagnostic only.',fill='white')
        for row,(person,_,cells) in enumerate(entries):
            y=55+row*205
            for column,cell in enumerate(cells):
                canvas.paste(tile(cell),(column*177+3,y))
            draw.text((3,y+180),person,fill='white')
        canvas.save(output/f'{mode}_reference_identity_comparison.jpg')
    canvas=Image.new('RGB',(1070,len(shape_pictures)*205+55),'#171717')
    draw=ImageDraw.Draw(canvas)
    for column,label in enumerate(['Clean','Input','Common shape GT','Common shape pred','Own shape GT','Own shape pred']):
        draw.text((column*177+3,8),label,fill='white')
    draw.text((3,30),'Normalized light-map grayscale range 0.75..1.25; same common geometry for columns 3/4.',fill='white')
    for row,(person,name,cells) in enumerate(shape_pictures):
        y=55+row*205
        for column,cell in enumerate(cells):
            canvas.paste(tile(cell),(column*177+3,y))
        draw.text((3,y+180),f'{person} / {name}',fill='white')
    canvas.save(output/'directional_light_shape_comparison.jpg')
    print(json.dumps({'summary':summaries,'interventions':intervention_summary}),flush=True)
    print('Saved:',output.resolve(),flush=True)


if __name__=='__main__':
    main()
