import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager as fm
fam=[f.name for f in fm.fontManager.ttflist if "CJK" in f.name and ("TC" in f.name or "SC" in f.name)]
plt.rcParams["font.family"]=fam[0] if fam else "sans-serif"
P,G,R,K="#7F77DD","#1D9E75","#D85A30","#444"
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5.6),gridspec_kw={"width_ratios":[1,1.15]})
C=np.array([0,0]);r=10
t=np.linspace(0,np.pi/2,200);a1.plot(r*np.cos(t),r*np.sin(t),"--",color="#999",lw=1,label="原本的圓（圓心 C）")
V=np.array([r*np.cos(np.pi/4),r*np.sin(np.pi/4)]);Vp=V+np.array([0.55,0.45])
def arc_through(p,q,rad):
    m=(p+q)/2;d=q-p;L=np.linalg.norm(d);h=np.sqrt(rad**2-(L/2)**2);n=np.array([-d[1],d[0]])/L
    cands=[m+h*n,m-h*n];c=min(cands,key=lambda c:np.linalg.norm(c))
    a=np.arctan2(p[1]-c[1],p[0]-c[0]);b=np.arctan2(q[1]-c[1],q[0]-c[0])
    while b-a>np.pi: b-=2*np.pi
    while a-b>np.pi: b+=2*np.pi
    tt=np.linspace(a,b,100);return c,np.c_[c[0]+rad*np.cos(tt),c[1]+rad*np.sin(tt)]
A=np.array([r,0]);B=np.array([0,r])
c1,arc1=arc_through(A,Vp,r);c2,arc2=arc_through(Vp,B,r)
a1.plot(*arc1.T,color=P,lw=3,label="第 1 段（穿過 A、V′）→ 圓心 C1");a1.plot(*arc2.T,color=G,lw=3,label="第 2 段（穿過 V′、B）→ 圓心 C2")
a1.plot([V[0],V[0]+5],[V[1],V[1]+3.5],color=K,lw=1.5,label="別的曲線，碰在 V")
a1.plot(*V,"o",mfc="white",mec=K,ms=7);a1.plot(*Vp,"o",color=R,ms=9,label="V′：V snap 到格點後（偏移放大）")
for c,col,n in ((C,K,"C"),(c1,P,"C1"),(c2,G,"C2")):
    a1.plot(*c,"o",color=col,ms=6);a1.annotate(n,c,xytext=(6,-12),textcoords="offset points",color=col,fontsize=11)
a1.annotate("V",V,xytext=(-16,-6),textcoords="offset points",fontsize=11);a1.annotate("V′",Vp,xytext=(6,4),textcoords="offset points",color=R,fontsize=11)
a1.annotate("A",A,xytext=(6,0),textcoords="offset points");a1.annotate("B",B,xytext=(0,6),textcoords="offset points")
a1.set_aspect("equal");a1.set_xlim(-2,22);a1.set_ylim(-2,13);a1.axis("off")
a1.set_title("1. 俯視：圓弧被切成兩段，兩段圓心不同",fontsize=13,loc="left")
a1.legend(loc="upper right",fontsize=9,frameon=False)
a2.set_title("2. 把側牆攤平（橫軸 = 沿弧長，縱軸 = Z）",fontsize=13,loc="left")
x0,w,h=0,6,3
a2.add_patch(Rectangle((x0,0),w,h,fill=False,ec="#999",lw=.8))
a2.plot([0,3.4],[0,0],color=P,lw=4);a2.plot([3.4,6],[0,0],color=G,lw=4);a2.plot([0,6],[h,h],color=K,lw=2)
a2.plot([0,0],[0,h],color=K,lw=2);a2.plot([6,6],[0,h],color=K,lw=2);a2.plot(3.4,0,"o",color=R,ms=9)
a2.text(3,1.5,"一片牆、5 條邊\n底邊兩段弧圓心不同\n→ 寫不成一個精確圓柱面\n→ writer 拒絕",ha="center",va="center",fontsize=11)
a2.text(3,-0.9,"現在",ha="center",fontsize=13,fontweight="bold")
a2.annotate("",xy=(7.6,1.5),xytext=(6.4,1.5),arrowprops=dict(arrowstyle="->",lw=1.5))
x1=8
a2.add_patch(Rectangle((x1,0),3.4,h,fc="#EEEDFE",ec=P,lw=1));a2.add_patch(Rectangle((x1+3.4,0),2.6,h,fc="#E1F5EE",ec=G,lw=1))
a2.plot([x1+3.4,x1+3.4],[0,h],color=R,lw=2.5);a2.plot([x1+3.4]*2,[0,h],"o",color=R,ms=8)
a2.text(x1+1.7,1.5,"圓柱 C1\n4 條邊",ha="center",va="center",fontsize=11,color="#3C3489")
a2.text(x1+4.7,1.5,"圓柱 C2\n4 條邊",ha="center",va="center",fontsize=11,color="#085041")
a2.text(x1+3,-0.9,"修法：在 V′ 加一條垂直邊，切成兩片",ha="center",fontsize=13,fontweight="bold")
a2.set_xlim(-0.5,14.5);a2.set_ylim(-1.6,4);a2.set_aspect("equal");a2.axis("off")
fig.tight_layout();fig.savefig("five_edge_wall.png",dpi=160,facecolor="white")
