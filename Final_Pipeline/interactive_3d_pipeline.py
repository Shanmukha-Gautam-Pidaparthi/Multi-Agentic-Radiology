#!/usr/bin/env python3
"""
interactive_3d_pipeline.py — 3D Clinical Tumor Pipeline (Fixed)
4 panels: CT Slice | Tumor UNet | TotalSeg | MedSAM
Controls: scroll=slice, draw box=MedSAM, R=3D render, P=report

Run:
  conda activate medsam_env && cd ~/Desktop/intern/Box_Prompt
  python interactive_3d_pipeline.py --nifti btcv/raw/data/kaggle_btcv/images/img0008-0002.nii
"""
import matplotlib; matplotlib.use('TkAgg')
import os, sys, json, argparse, glob, torch, numpy as np, nibabel as nib
import torch.nn.functional as F, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from scipy.ndimage import binary_dilation
from PIL import Image
from datetime import datetime
from segment_anything.build_sam import _build_sam

NUM_CLS=11
CLS={0:"bg",1:"liver",2:"liver_tumor",3:"kidney",4:"kidney_tumor",
     5:"pancreas",6:"pancreas_tumor",7:"colon",8:"colon_tumor",9:"esophagus",10:"esophagus_tumor"}
TUMOR_IDS={2,4,6,8,10}; ORGAN_IDS={1,3,5,7,9}
COLORS={1:[255,200,50],2:[255,30,30],3:[255,140,0],4:[255,100,0],
        5:[255,255,100],6:[200,200,0],7:[100,255,100],8:[0,200,50],9:[180,100,255],10:[150,0,200]}
BTCV={1:"spleen",2:"right_kidney",3:"left_kidney",4:"gallbladder",5:"esophagus",
      6:"liver",7:"stomach",8:"aorta",9:"IVC",10:"portal_vein",11:"pancreas",12:"R_adrenal",13:"L_adrenal"}
TS2B={"spleen":1,"kidney_right":2,"kidney_left":3,"gallbladder":4,"esophagus":5,
      "liver":6,"stomach":7,"aorta":8,"inferior_vena_cava":9,
      "portal_vein_and_splenic_vein":10,"pancreas":11,"adrenal_gland_right":12,"adrenal_gland_left":13}

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--nifti",required=True); p.add_argument("--device",default="cpu")
    p.add_argument("--checkpoint",default="best_medsam_btcv.pth")
    p.add_argument("--save-dir",default="pipeline_3d_outputs"); return p.parse_args()

def load_unet(path,dev):
    from monai.networks.nets import UNet
    m=UNet(spatial_dims=3,in_channels=1,out_channels=NUM_CLS,channels=(32,64,128,256),strides=(2,2,2),num_res_units=2)
    st=torch.load(path,map_location="cpu",weights_only=False)
    m.load_state_dict(st if not isinstance(st,dict) or "model_state_dict" not in st else st["model_state_dict"])
    m.to(dev).eval(); return m

def load_segresnet(path,cfg_path,dev):
    from monai.networks.nets import SegResNet
    cfg=json.load(open(cfg_path))
    m=SegResNet(spatial_dims=2,in_channels=1,out_channels=cfg["NUM_CLASSES"],
                init_filters=16,blocks_down=(1,2,2,4),blocks_up=(1,1,1),dropout_prob=0.2)
    m.load_state_dict(torch.load(path,map_location="cpu",weights_only=False))
    m.to(dev).eval(); return m,cfg

def load_medsam(path,dev):
    sam=_build_sam(encoder_embed_dim=768,encoder_depth=12,encoder_num_heads=12,
                   encoder_global_attn_indexes=[2,5,8,11],checkpoint=None)
    ck=torch.load(path,map_location="cpu",weights_only=False)
    sam.load_state_dict(ck.get("model_state_dict",ck)); sam.to(dev).eval(); return sam

@torch.no_grad()
def run_3d(model,vol,dev):
    from monai.inferers import sliding_window_inference
    d=np.clip(vol.astype(np.float32),-175,250); d=(d+175)/425.0
    t=torch.from_numpy(d).unsqueeze(0).unsqueeze(0).float().to(dev)
    p=sliding_window_inference(t,(96,96,96),1,model,overlap=0.25,mode="gaussian")
    return torch.argmax(p,dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

def load_ts(case):
    masks={}
    for d in [f"totalseg_cache/{case}","totalseg_test_output"]:
        if os.path.isdir(d):
            for n,bid in TS2B.items():
                p=os.path.join(d,f"{n}.nii.gz")
                if os.path.exists(p):
                    a=nib.load(p).get_fdata()
                    if a.sum()>0: masks[bid]=a
            if masks: break
    return masks

def get_slice(vol,z):
    s=np.clip(vol[:,:,z].astype(np.float32),-175,250)
    return np.flip(((s+175)/425*255).astype(np.uint8),axis=0).copy()

def overlay(img,pred_s):
    rgb=np.stack([img]*3,axis=-1).astype(np.float32)
    for cid,col in COLORS.items():
        m=pred_s==cid
        if not m.any(): continue
        a=0.6 if cid in TUMOR_IDS else 0.25
        for c in range(3): rgb[:,:,c]=np.where(m,(1-a)*rgb[:,:,c]+a*col[c],rgb[:,:,c])
    return np.clip(rgb,0,255).astype(np.uint8)

def classify_box(ts_masks,si,box,orig_shape,img_shape):
    if not ts_masks or si is None: return "unknown",[]
    oh,ow=orig_shape[:2]; sh,sw=img_shape[0]/oh,img_shape[1]/ow
    bx1,by1,bx2,by2=int(box[0]),int(box[1]),int(box[2]),int(box[3])
    ox1,oy1,ox2,oy2=int(bx1/sw),int(by1/sh),int(bx2/sw),int(by2/sh)
    res=[]
    for bid,m3d in ts_masks.items():
        best=0
        for z in range(max(0,si-15),min(m3d.shape[2],si+16)):
            ms=np.flip(m3d[:,:,z],axis=0).copy()
            v=int(ms[oy1:oy2,ox1:ox2].sum())
            if v>best: best=v
        if best==0: continue
        tot=max(1,(oy2-oy1)*(ox2-ox1))
        res.append({"bid":bid,"name":BTCV.get(bid,f"org_{bid}"),"vox":best,"pct":round(100*best/tot,2)})
    res.sort(key=lambda x:-x["vox"])
    return (res[0]["name"] if res else "unknown"),res

def calc_vols(pred,pixdim):
    vv=float(np.prod(pixdim[:3])); r={}
    for cid in sorted(CLS.keys()):
        if cid==0: continue
        cnt=int((pred==cid).sum())
        if cnt>0: r[CLS[cid]]={"vox":cnt,"mm3":round(cnt*vv,1),"cm3":round(cnt*vv/1000,2),"tumor":cid in TUMOR_IDS}
    return r

def gen_report(vols,case,sp):
    lines=["="*60,"  CLINICAL DIAGNOSTIC REPORT","="*60,
           f"  Case: {case}",f"  Date: {datetime.now():%Y-%m-%d %H:%M}","","  FINDINGS:","-"*40]
    tumors=[]
    for n,i in sorted(vols.items()):
        t="**TUMOR**" if i["tumor"] else "organ"
        lines.append(f"    {n:>18}: {i['cm3']:>8.2f} cm3 ({i['vox']:>8,} vox) [{t}]")
        if i["tumor"]: tumors.append(f"{n} ({i['cm3']:.2f} cm3)")
    lines+=["","-"*40,"  SUMMARY:"]
    if tumors:
        lines.append(f"    TUMORS: {len(tumors)}")
        for t in tumors: lines.append(f"      - {t}")
    else: lines.append("    No tumors detected.")
    lines+=["","="*60]
    rpt="\n".join(lines)
    with open(sp,"w") as f: f.write(rpt)
    return rpt

def render_3d(pred,medsam_3d,pixdim,case,save_dir):
    print("\n  Rendering 3D mesh...")
    fig=plt.figure(figsize=(16,12),facecolor="#0a0a0a")
    ax=fig.add_subplot(111,projection='3d',facecolor="#0a0a0a")
    rendered=False
    for cid in sorted(TUMOR_IDS):
        m=(pred==cid)
        if m.sum()<10: continue
        md=binary_dilation(m,iterations=2)
        try:
            v,f,_,_=marching_cubes(md.astype(float),level=0.5,spacing=pixdim[:3])
            col=[c/255 for c in COLORS[cid]]
            mesh=Poly3DCollection(v[f],alpha=0.7,edgecolor=col,linewidth=0.1)
            mesh.set_facecolor(col+[0.6]); ax.add_collection3d(mesh)
            ax.auto_scale_xyz(v[:,0],v[:,1],v[:,2])
            print(f"    {CLS[cid]}: {len(f)} faces"); rendered=True
        except: pass
    for cid in sorted(ORGAN_IDS):
        m=(pred==cid)
        if m.sum()<500: continue
        ds=m[::4,::4,::4]
        try:
            sp_=tuple(p*4 for p in pixdim[:3])
            v,f,_,_=marching_cubes(ds.astype(float),level=0.5,spacing=sp_)
            col=[c/255 for c in COLORS[cid]]
            mesh=Poly3DCollection(v[f],alpha=0.15,edgecolor=col,linewidth=0.02)
            mesh.set_facecolor(col+[0.1]); ax.add_collection3d(mesh)
            ax.auto_scale_xyz(v[:,0],v[:,1],v[:,2]); rendered=True
        except: pass
    if medsam_3d.sum()>10:
        try:
            md=binary_dilation(medsam_3d>0,iterations=2)
            v,f,_,_=marching_cubes(md.astype(float),level=0.5,spacing=pixdim[:3])
            mesh=Poly3DCollection(v[f],alpha=0.5,edgecolor=[0,1,0.5],linewidth=0.1)
            mesh.set_facecolor([0,1,0.5,0.4]); ax.add_collection3d(mesh)
            ax.auto_scale_xyz(v[:,0],v[:,1],v[:,2])
            print(f"    MedSAM: {len(f)} faces"); rendered=True
        except: pass
    if not rendered:
        ax.text2D(0.5,0.5,"No structures to render",transform=ax.transAxes,ha="center",color="white",fontsize=14)
    ax.set_xlabel("X mm",color="white"); ax.set_ylabel("Y mm",color="white"); ax.set_zlabel("Z mm",color="white")
    ax.tick_params(colors="white"); ax.set_title(f"3D Tumor Mesh - {case}",color="white",fontsize=14,fontweight="bold")
    ax.view_init(elev=30,azim=45)
    p=os.path.join(save_dir,f"3d_{case}.png")
    fig.savefig(p,dpi=150,bbox_inches="tight",facecolor="#0a0a0a"); print(f"  Saved: {p}"); plt.show()

def main():
    args=parse_args(); dev=torch.device(args.device); os.makedirs(args.save_dir,exist_ok=True)
    print("\n"+"="*60+"\n  3D INTERACTIVE CLINICAL TUMOR PIPELINE\n"+"="*60)
    nii=nib.load(args.nifti); vol=nii.get_fdata(); pixdim=nii.header.get_zooms()
    H,W,D=vol.shape; case=os.path.basename(args.nifti).replace(".nii.gz","").replace(".nii","")
    print(f"  Volume: {vol.shape} | Spacing: {pixdim[:3]}")

    print("\n  Loading models...")
    unet=load_unet("weights/best_multiorgan.pth",dev)
    segresnet,seg_cfg=load_segresnet("SegResNet_Project/best_segresnet_model.pth","SegResNet_Project/model_config.json",dev)
    sam=load_medsam(args.checkpoint,dev)
    ts_masks=load_ts(case); print(f"    TotalSeg: {len(ts_masks)} organs")

    print("\n  Running 3D UNet (1-3 min on CPU)...")
    pred_3d=run_3d(unet,vol,dev)
    vols=calc_vols(pred_3d,pixdim)
    for n,i in sorted(vols.items()):
        print(f"    {n:>18}: {i['cm3']:>6.2f} cm3 [{'TUMOR' if i['tumor'] else 'organ'}]")

    rpt_path=os.path.join(args.save_dir,f"report_{case}.txt")
    rpt=gen_report(vols,case,rpt_path); print(f"  Report: {rpt_path}")

    # ── 4-Panel Figure ──
    fig=plt.figure(figsize=(28,8),facecolor="#0a0a0a")
    fig.suptitle("3D CLINICAL PIPELINE | Scroll=slice | Draw box=segment | R=3D | P=report",
                 color="white",fontsize=12,fontweight="bold")
    ax1=fig.add_axes([0.01,0.13,0.24,0.80])  # CT
    ax2=fig.add_axes([0.26,0.13,0.24,0.80])  # Tumor UNet
    ax3=fig.add_axes([0.51,0.13,0.22,0.80])  # TotalSeg
    ax4=fig.add_axes([0.74,0.13,0.25,0.80])  # MedSAM
    ax_sl=fig.add_axes([0.01,0.03,0.48,0.04])
    for a in [ax1,ax2,ax3,ax4]: a.set_facecolor("#0a0a0a"); a.axis("off")
    ax3.set_facecolor("#1a1a2e")

    slider=Slider(ax_sl,'Slice',0,D-1,valinit=D//2,valstep=1,color="#00FF88",initcolor="none")
    slider.label.set_color("white"); slider.valtext.set_color("white")

    st={"z":D//2,"drawing":False,"start":None,"rect":None,"emb":None,"emb_z":None,
        "mseg":np.zeros((H,W,D),dtype=np.uint8),"last_mask":None,"last_box":None,"last_score":0}

    def draw(z):
        st["z"]=int(z); ct=get_slice(vol,st["z"]); ps=np.flip(pred_3d[:,:,st["z"]],axis=0).copy()
        ax1.clear(); ax1.imshow(ct,cmap="gray"); ax1.set_title(f"CT Slice {st['z']}/{D-1} (draw box)",color="white",fontsize=11)
        ax1.axis("off"); ax1.set_facecolor("#0a0a0a")
        ax2.clear(); ax2.imshow(overlay(ct,ps)); ax2.axis("off"); ax2.set_facecolor("#0a0a0a")
        found=[CLS[c] for c in range(1,NUM_CLS) if (ps==c).any()]
        has_t=any(c in TUMOR_IDS for c in range(1,NUM_CLS) if (ps==c).any())
        tc="#FF5252" if has_t else "#4CAF50"
        ax2.set_title("UNet: "+(", ".join(found[:4]) if found else "bg"),color=tc,fontsize=10,fontweight="bold")
        # Keep MedSAM panel if same slice
        if st["last_mask"] is not None and st["last_box"] is not None:
            show_medsam(ct)
        else:
            ax4.clear(); ax4.imshow(ct,cmap="gray"); ax4.axis("off"); ax4.set_facecolor("#0a0a0a")
            ax4.set_title("MedSAM (draw box on left)",color="white",fontsize=11)
        fig.canvas.draw_idle()

    def show_medsam(ct):
        ax4.clear(); ax4.set_facecolor("#0a0a0a"); ax4.axis("off")
        rgb=np.stack([ct]*3,axis=-1)
        ax4.imshow(rgb)
        m=st["last_mask"]; b=st["last_box"]; sc=st["last_score"]
        ov=np.zeros((*m.shape[:2],4),dtype=np.uint8)
        ov[m>0]=[0,255,100,170]
        ax4.imshow(ov)
        if m.any(): ax4.contour(m,levels=[0.5],colors=["yellow"],linewidths=[2])
        ax4.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],lw=2,edgecolor="cyan",facecolor="none",linestyle="--"))
        ax4.set_title(f"MedSAM: {m.sum():,}px | score={sc:.3f}",color="#00FF88",fontsize=11,fontweight="bold")

    draw(D//2)

    def on_slider(v): st["last_mask"]=None; st["last_box"]=None; draw(int(v))
    slider.on_changed(on_slider)

    def on_scroll(e):
        nz=min(D-1,st["z"]+1) if e.button=='up' else max(0,st["z"]-1)
        slider.set_val(nz)

    def on_press(e):
        if e.inaxes!=ax1 or e.xdata is None: return
        st["drawing"]=True; st["start"]=(e.xdata,e.ydata)

    def on_motion(e):
        if not st["drawing"] or e.inaxes!=ax1 or e.xdata is None: return
        x0,y0=st["start"]; x1,y1=e.xdata,e.ydata
        if st["rect"]: st["rect"].remove()
        st["rect"]=ax1.add_patch(Rectangle((min(x0,x1),min(y0,y1)),abs(x1-x0),abs(y1-y0),
            lw=2,edgecolor="#00FF88",facecolor="#00FF88",alpha=0.15,linestyle="--"))
        fig.canvas.draw_idle()

    @torch.no_grad()
    def on_release(e):
        if not st["drawing"]: return
        st["drawing"]=False
        if e.inaxes!=ax1 or e.xdata is None: return
        x0,y0=st["start"]; x1,y1=e.xdata,e.ydata
        if abs(x1-x0)<5 or abs(y1-y0)<5: return
        box=[min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)]
        ct=get_slice(vol,st["z"]); rgb=np.stack([ct]*3,axis=-1); ih,iw=rgb.shape[:2]

        # MedSAM embedding
        if st["emb_z"]!=st["z"]:
            print(f"  Computing MedSAM embedding (slice {st['z']})...")
            img_t=torch.from_numpy(rgb).float().permute(2,0,1).to(dev)
            st["emb"]=sam.image_encoder(sam.preprocess(img_t).unsqueeze(0)); st["emb_z"]=st["z"]

        sx,sy=1024.0/iw,1024.0/ih
        b=np.array(box,dtype=np.float32); b1=b.copy(); b1[0]*=sx; b1[1]*=sy; b1[2]*=sx; b1[3]*=sy
        bt=torch.from_numpy(b1).float().to(dev).unsqueeze(0).unsqueeze(0)
        sp,dn=sam.prompt_encoder(points=None,boxes=bt,masks=None)
        lr,iou=sam.mask_decoder(image_embeddings=st["emb"],image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sp,dense_prompt_embeddings=dn,multimask_output=True)
        ms=F.interpolate(lr,(1024,1024),mode="bilinear",align_corners=False).squeeze(0)
        bi=torch.argmax(iou.squeeze(0)).item()
        p1=(torch.sigmoid(ms[bi])>0.5).cpu().numpy().astype(np.uint8)
        mask=(np.array(Image.fromarray(p1*255).resize((iw,ih),Image.NEAREST))>127).astype(np.uint8)
        score=iou.squeeze(0)[bi].item()

        st["last_mask"]=mask; st["last_box"]=box; st["last_score"]=score
        st["mseg"][:,:,st["z"]]=np.maximum(st["mseg"][:,:,st["z"]],np.flip(mask,axis=0).copy())

        # TotalSeg classification
        winner,ts_res=classify_box(ts_masks,st["z"],box,(H,W),rgb.shape[:2])
        print(f"  Slice {st['z']}: TotalSeg={winner} | MedSAM={mask.sum():,}px score={score:.3f}")

        # Update TotalSeg panel
        ax3.clear(); ax3.set_facecolor("#1a1a2e")
        if ts_res:
            top=ts_res[:6]; names=[r["name"] for r in top]; pcts=[r["pct"] for r in top]
            cols=["#4CAF50" if i==0 else "#2196F3" if i<3 else "#607D8B" for i in range(len(top))]
            bars=ax3.barh(names[::-1],pcts[::-1],color=cols[::-1],height=0.55)
            for bar,val in zip(bars,pcts[::-1]):
                ax3.text(min(val+0.5,max(pcts)*0.95 if max(pcts)>0 else 1),
                         bar.get_y()+bar.get_height()/2,f"{val:.1f}%",va="center",color="white",fontsize=9)
            ax3.set_title(f"TotalSeg: {winner.upper()}",color="#4CAF50",fontsize=12,fontweight="bold")
            ax3.tick_params(colors="white"); ax3.spines[:].set_color("#333")
            for l in ax3.get_xticklabels()+ax3.get_yticklabels(): l.set_color("white")
        else:
            ax3.text(0.5,0.5,"No TotalSeg masks\nRun step1_precompute_totalseg.py",
                     transform=ax3.transAxes,ha="center",va="center",color="#FF5252",fontsize=11)
            ax3.axis("off")

        # Update MedSAM panel
        show_medsam(ct)

        # Redraw CT with box
        ax1.clear(); ax1.imshow(ct,cmap="gray"); ax1.axis("off"); ax1.set_facecolor("#0a0a0a")
        ax1.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
            lw=2.5,edgecolor="#00FF88",facecolor="none",linestyle="--"))
        ax1.set_title(f"CT Slice {st['z']} | Box drawn",color="#00FF88",fontsize=11)
        fig.canvas.draw_idle()

        fp=os.path.join(args.save_dir,f"s{st['z']:04d}.png")
        fig.savefig(fp,dpi=100,bbox_inches="tight",facecolor="#0a0a0a"); print(f"  Saved: {fp}")

    def on_key(e):
        if e.key=='r': render_3d(pred_3d,st["mseg"],pixdim,case,args.save_dir)
        elif e.key=='p': print("\n"+rpt)

    fig.canvas.mpl_connect("scroll_event",on_scroll)
    fig.canvas.mpl_connect("button_press_event",on_press)
    fig.canvas.mpl_connect("motion_notify_event",on_motion)
    fig.canvas.mpl_connect("button_release_event",on_release)
    fig.canvas.mpl_connect("key_press_event",on_key)

    print("\n"+"="*60)
    print("  Scroll=navigate | Draw box=MedSAM | R=3D mesh | P=report")
    print("="*60+"\n")
    plt.show()

if __name__=="__main__":
    main()
