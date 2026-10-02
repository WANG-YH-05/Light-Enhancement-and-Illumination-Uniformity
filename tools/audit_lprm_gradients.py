"""Read-only LPRM parameter/gradient audit on deterministic physical groups."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rrnet.config import load_config, model_kwargs
from rrnet.losses import luminance
from rrnet.physical_reference_data import MEADPhysicalReferencePairs
from rrnet.reference_relative_model import ReferenceRelativeRRNet
from rrnet.reference_relative_loss import ReferenceRelativeLoss
from train import model_prediction


def mean(x, mask):
    return (x * mask).sum() / (x.shape[1] * mask.sum().clamp_min(1))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True)
    args=p.parse_args()
    c=load_config(args.config)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=ReferenceRelativeRRNet(**model_kwargs(c,args.config)).to(device).eval()
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint.get('model',checkpoint))
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    criterion=ReferenceRelativeLoss(**c['loss']).to(device)
    d=c['data']
    rows=[]
    for split in ('train','val'):
        ds=MEADPhysicalReferencePairs(d['root'],split,metadata_file=d.get('metadata_file','metadata.csv'),
            seed=20261001,num_lights=c['model']['num_lights'],variants_per_source=1,
            dynamic_epoch=False,group_size=4,reference_sensitivity_fraction=0,
            reference_lighting_weights={'normal':1})
        for person in sorted(set(ds.people))[:3]:
            batch=default_collate([ds[ds.people.index(person)]])
            source,target,mask,skin,pred,extras=model_prediction(model,batch,device,True,True,True)
            loss=criterion(pred,target,source,mask,skin,**extras)
            # Grouped normalized/physical theta are parallel expanded views.
            # Map losses descend from physical theta, not the parallel view.
            physical_theta=pred['source_theta']
            shape_gradient,=torch.autograd.grad(loss['illum_shape'],physical_theta,retain_graph=True)
            map_gradient,=torch.autograd.grad(loss['illum_source'],physical_theta,retain_graph=True)
            full_gradient,=torch.autograd.grad(loss['total'],physical_theta,retain_graph=True)
            head_gradient,coarse_gradient=torch.autograd.grad(
                loss['illum_shape'],(model.base.lprm.r1.weight,model.base.lprm.r0[1].weight))
            raw=pred['source_theta'].detach()
            physical=model.renderer.physical_parameters(raw)
            face=extras['source_light_mask']
            light=luminance(pred['source_illumination'].detach())
            ambient=luminance(physical['ambient'][:,:,None,None])
            ambient_ratio=float(mean(ambient.expand_as(light)/light.clamp_min(1e-5),face))
            true_light,_=model.illumination_from_theta(pred['depth'].detach(),extras['source_theta_target'])
            true_params=model.renderer.physical_parameters(extras['source_theta_target'])
            true_ambient=luminance(true_params['ambient'][:,:,None,None])
            true_ratio=float(mean(true_ambient.expand_as(light)/luminance(true_light).clamp_min(1e-5),face))
            detail={}
            for name,sl in [('color',slice(0,3)),('direction',slice(3,6)),('position',slice(6,9)),('attenuation',slice(9,10))]:
                values=raw[:,:-3].reshape(-1,c['model']['num_lights'],10)[...,sl]
                mg=map_gradient[:,:-3].reshape(-1,c['model']['num_lights'],10)[...,sl]
                sg=shape_gradient[:,:-3].reshape(-1,c['model']['num_lights'],10)[...,sl]
                fg=full_gradient[:,:-3].reshape(-1,c['model']['num_lights'],10)[...,sl]
                detail[name]=dict(min=float(values.min()),max=float(values.max()),
                    negative_fraction=float((values<0).float().mean()),
                    outside_unit_fraction=float(((values<0)|(values>1)).float().mean()),
                    map_zero_gradient_fraction=float((mg.abs()<1e-12).float().mean()),
                    shape_mean_abs_gradient=float(sg.abs().mean()),
                    render_branch_mean_abs_gradient=float(fg.abs().mean()))
            rows.append(dict(split=split,person=person,predicted_ambient_share=ambient_ratio,
                target_ambient_share=true_ratio,parameter_fields=detail,
                fully_off_light_fraction=float((physical['color'].sum(-1)==0).float().mean()),
                shape_head_gradient_norm=float(head_gradient.norm()),
                shape_coarse_head_gradient_norm=float(coarse_gradient.norm()),
                losses={k:float(loss[k].detach()) for k in ('total','illum_shape','illum_source','theta_source')}))
            print(json.dumps(rows[-1]),flush=True)
    assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items()),'Audit modified model state'
    report=dict(checkpoint=str(Path(args.checkpoint).resolve()),rows=rows,
        denormalizer_mean=model.base.lprm.denormalize.mean.detach().cpu().tolist(),
        denormalizer_std=model.base.lprm.denormalize.std.detach().cpu().tolist(),
        scope='Six physical groups, no updates. Gradients wrt physical source theta. Total physical-view gradient excludes the parallel direct-normalized-theta-supervision path; actual head gradient proves end-to-end connectivity.')
    output=Path('outputs/lprm_code_audit')/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    (output/'gradient_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Saved:',output.resolve(),flush=True)


if __name__=='__main__':
    main()
