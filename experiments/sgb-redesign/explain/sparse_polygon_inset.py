import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from shapely.geometry import Polygon, Point, box
fam=[f.name for f in fm.fontManager.ttflist if "CJK" in f.name and ("TC" in f.name or "SC" in f.name)]
plt.rcParams["font.family"]=fam[0] if fam else "sans-serif"
r,m,L=10,3,22
def fill(ax,geom,**kw):
    geoms=getattr(geom,"geoms",[geom])
    for g in geoms:
        x,y=g.exterior.xy;ax.fill(x,y,**kw)
        for h in g.interiors:
            x,y=h.xy;ax.fill(x,y,color="white")
fig,axes=plt.subplots(1,2,figsize=(13,6.6))
cases=[("宣告成真正的圓",Point(0,0).buffer(r,resolution=256)),("直接用 layout 的 16 邊形",Polygon([(r*np.cos(t),r*np.sin(t)) for t in np.linspace(0,2*np.pi,17)[:-1]+np.pi/16]))]
for ax,(title,hole) in zip(axes,cases):
    metal=box(-L,-L,L,L).difference(hole)
    inset_edge=hole.buffer(m,resolution=256)
    band=inset_edge.difference(hole)
    fill(ax,metal,color="#D3D1C7")
    fill(ax,band,color="#F5C4B3")
    x,y=hole.exterior.xy;ax.plot(x,y,color="#333",lw=2)
    x,y=inset_edge.exterior.xy;ax.plot(x,y,color="#D85A30",lw=2)
    ax.plot(*Point(0,0).buffer(r+m,resolution=256).exterior.xy,"--",color="#185FA5",lw=1)
    ax.set_aspect("equal");ax.set_xlim(-16,16);ax.set_ylim(-16,16);ax.axis("off");ax.set_title(title,fontsize=14)
axes[0].text(0,-15.2,"內縮邊界（紅）正好是半徑 r+m 的圓（藍虛線）",ha="center",fontsize=11)
axes[1].text(0,-15.2,"內縮邊界（紅）＝16 段直線＋每個角一小段半徑 m 的圓弧\n和真正的圓（藍虛線）有一點差距",ha="center",fontsize=11)
v=np.array([r*np.cos(np.pi/16),r*np.sin(np.pi/16)])
axes[1].annotate("多邊形的角：\n內縮後變成一小段圓弧",xy=(v[0]+m*.98,v[1]+m*.2),xytext=(11.5,9),fontsize=10,arrowprops=dict(arrowstyle="->",lw=1))
for ax in axes:
    ax.text(0,0,"孔\n(vacuum)",ha="center",va="center",fontsize=11)
    ax.text(-13,13,"金屬",fontsize=12)
fig.text(.5,.02,"灰：金屬　橘：離孔邊緣不到 m 的那一圈（inset band）　（m 刻意畫大，方便看清楚）",ha="center",fontsize=11)
fig.tight_layout(rect=(0,.05,1,.94));fig.savefig("sparse_polygon_inset.png",dpi=150,facecolor="white")
