import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager as fm
fam=[f.name for f in fm.fontManager.ttflist if "CJK" in f.name and ("TC" in f.name or "SC" in f.name)]
plt.rcParams["font.family"]=fam[0] if fam else "sans-serif"
A,B,J,R="#85B7EB","#97C459","#F0997B","#E24B4A"
fig,axes=plt.subplots(3,1,figsize=(11,11.5))
def rect(ax,x0,x1,y0,y1,fc,ec="#333",lw=1,alpha=1,hatch=None,ls="-"):
    ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fc=fc,ec=ec,lw=lw,alpha=alpha,hatch=hatch,ls=ls))
def frame(ax,title):
    ax.set_xlim(-0.5,12.5);ax.set_ylim(-2.2,1.8);ax.set_aspect("equal");ax.axis("off");ax.set_title(title,fontsize=13,loc="left")
# 1 authored
ax=axes[0];frame(ax,"1. 使用者畫的（模擬用 junction 一定會和兩邊的電極重疊）")
rect(ax,0,4,-1,1,A);rect(ax,8,12,-1,1,B)
rect(ax,3,9,-0.6,0.6,J,alpha=.75)
rect(ax,3,4,-0.6,0.6,"none",ec="#7a1f1f",lw=1.5,hatch="///");rect(ax,8,9,-0.6,0.6,"none",ec="#7a1f1f",lw=1.5,hatch="///")
ax.text(1.5,0,"電極 A\n(Net A)",ha="center",va="center",fontsize=11);ax.text(10.5,0,"電極 B\n(Net B)",ha="center",va="center",fontsize=11)
ax.text(6,0,"junction sheet S",ha="center",va="center",fontsize=11)
ax.text(3.5,-1.55,"重疊",ha="center",fontsize=10,color="#7a1f1f");ax.text(8.5,-1.55,"重疊",ha="center",fontsize=10,color="#7a1f1f")
# 2 S1 partition
ax=axes[1];frame(ax,"2. S1 的 junction 切割（現行 develop 已有、新 SGB 直接沿用）")
rect(ax,0,4,-1,1,A);rect(ax,8,12,-1,1,B)
rect(ax,4,5,-0.6,0.6,A,ec="#185FA5",lw=1.5);rect(ax,7,8,-0.6,0.6,B,ec="#3B6D11",lw=1.5)
rect(ax,5,7,-0.6,0.6,R,ec="#7a1f1f",lw=1.5,alpha=.85)
ax.text(4.5,0,"A 端",ha="center",va="center",fontsize=10);ax.text(7.5,0,"B 端",ha="center",va="center",fontsize=10)
ax.text(6,0,"R\nRLC 矩形",ha="center",va="center",fontsize=10,color="white")
ax.text(6,-1.6,"① 先去掉和電極重疊的部分：E = S − (電極 A ∪ 電極 B)\n② 再切成三段：A 端（併入 Net A）、中間矩形 R（lumped RLC）、B 端（併入 Net B）\n　三段剛好鋪滿 E、彼此不重疊，也不和電極重疊",ha="center",va="top",fontsize=10)
# 3 new SGB
ax=axes[2];frame(ax,"3. 新 SGB 收到的、以及它做的事")
from matplotlib.patches import Polygon
ax.add_patch(Polygon([(0,-1),(4,-1),(4,-0.6),(5,-0.6),(5,0.6),(4,0.6),(4,1),(0,1)],fc=A,ec="#185FA5",lw=1.5))
ax.add_patch(Polygon([(12,-1),(8,-1),(8,-0.6),(7,-0.6),(7,0.6),(8,0.6),(8,1),(12,1)],fc=B,ec="#3B6D11",lw=1.5))
for x in (4,8):
    ax.plot([x,x],[-0.6,0.6],ls=":",color="#555",lw=1.2)
rect(ax,5,7,-0.6,0.6,R,ec="#7a1f1f",lw=1.5,alpha=.85)
ax.text(2,0,"電極 A ＋ A 端\n（同一個 Entity）",ha="center",va="center",fontsize=10)
ax.text(10,0,"電極 B ＋ B 端\n（同一個 Entity）",ha="center",va="center",fontsize=10)
ax.text(6,0,"PORT",ha="center",va="center",fontsize=10,color="white")
ax.text(6,-1.6,"・虛線：A 端與電極 A 之間的接縫，同一個 Entity → dissolution 自動溶掉，不留邊\n・R 成為 lumped-port sheet，和兩端共用精確的邊 → conformal；Route A 在 sheet 平面、Route B 在金屬中間平面\n・全程沒有重疊，不需要「挑一個」；若輸入真的有無法解析的重疊，直接報錯",ha="center",va="top",fontsize=10)
fig.tight_layout();fig.savefig("junction.png",dpi=150,facecolor="white")
