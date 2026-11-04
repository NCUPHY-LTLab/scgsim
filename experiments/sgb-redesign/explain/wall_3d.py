import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fam=[f.name for f in fm.fontManager.ttflist if "CJK" in f.name and ("TC" in f.name or "SC" in f.name)]
plt.rcParams["font.family"]=fam[0] if fam else "sans-serif"
P,G,R,K="#7F77DD","#1D9E75","#D85A30","#333"
r,h=10,4
def arc(c,p,q,n=60):
    a=np.arctan2(p[1]-c[1],p[0]-c[0]);b=np.arctan2(q[1]-c[1],q[0]-c[0])
    while b-a>np.pi:b-=2*np.pi
    while a-b>np.pi:b+=2*np.pi
    t=np.linspace(a,b,n);return np.c_[c[0]+r*np.cos(t),c[1]+r*np.sin(t)]
def center(p,q):
    m=(p+q)/2;d=q-p;L=np.linalg.norm(d);hh=np.sqrt(r**2-(L/2)**2);nn=np.array([-d[1],d[0]])/L
    return min([m+hh*nn,m-hh*nn],key=np.linalg.norm)
A=np.array([r,0.]);B=np.array([0.,r]);V=np.array([r/np.sqrt(2)]*2);Vp=V+np.array([.55,.45])
c1,c2=center(A,Vp),center(Vp,B);a1,a2=arc(c1,A,Vp),arc(c2,Vp,B)
def panel(ax,pts,col,alpha):
    quads=[[ (*pts[i],0),(*pts[i+1],0),(*pts[i+1],h),(*pts[i],h)] for i in range(len(pts)-1)]
    ax.add_collection3d(Poly3DCollection(quads,facecolor=col,edgecolor="none",alpha=alpha))
fig=plt.figure(figsize=(13,6))
for k,title in enumerate(["現在：一整片彎曲的牆（writer 拒絕）","修法：在 V′ 豎一條直線，把牆切成兩片"]):
    ax=fig.add_subplot(1,2,k+1,projection="3d")
    full=np.r_[a1,a2[1:]]
    if k==0:
        panel(ax,full,"#D3D1C7",.55)
        ax.plot(*a1.T,0,color=P,lw=3);ax.plot(*a2.T,0,color=G,lw=3)
        top=arc(np.zeros(2),A,B,120);ax.plot(*top.T,h,color=K,lw=2)
        ax.text2D(.02,.02,"底邊：兩段弧（紫、綠）圓心不同\n頂邊：只有一段弧\n→ 這片牆不是「一個」圓柱面，寫不出來",transform=ax.transAxes,fontsize=11)
    else:
        panel(ax,a1,"#AFA9EC",.6);panel(ax,a2,"#9FE1CB",.6)
        ax.plot(*a1.T,0,color=P,lw=3);ax.plot(*a2.T,0,color=G,lw=3)
        ax.plot(*a1.T,h,color=P,lw=3);ax.plot(*a2.T,h,color=G,lw=3)
        ax.plot([Vp[0]]*2,[Vp[1]]*2,[0,h],color=R,lw=3)
        ax.text2D(.02,.02,"紫色那片：沿圓心 C1 彎（精確圓柱）\n綠色那片：沿圓心 C2 彎（精確圓柱）\n兩片共用中間那條紅色直線，接縫不會裂開",transform=ax.transAxes,fontsize=11)
    ax.plot([Vp[0]],[Vp[1]],[0],"o",color=R,ms=8);ax.text(Vp[0]+.6,Vp[1]+.6,0,"V′",color=R,fontsize=12)
    ax.text(*A,-.4,"A");ax.text(*B,-.4,"B")
    ax.set_title(title,fontsize=13);ax.set_box_aspect((1,1,.45));ax.view_init(elev=22,azim=-40)
    ax.set_xlim(0,11);ax.set_ylim(0,11);ax.set_zlim(0,h);ax.set_axis_off()
fig.tight_layout();fig.savefig("wall_3d.png",dpi=150,facecolor="white")
