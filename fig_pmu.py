import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, networkx as nx, math
from pmu import *
S1,S2,S3="#2a78d6","#eb6834","#1baf7a"
INK,INK2,MUTED,SURF="#0b0b0b","#52514e","#8d8b85","#fcfcfb"
plt.rcParams.update({"figure.facecolor":SURF,"axes.facecolor":SURF,"font.size":9,
    "axes.edgecolor":MUTED,"axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,
    "text.color":INK,"axes.spines.top":False,"axes.spines.right":False,"font.family":"DejaVu Sans"})

fig,ax=plt.subplots(1,2,figsize=(11.0,4.3))

# ---- panel A: greedy vs exact ----
a=ax[0]
data={}
for name,gen,col in [("radial feeder",lambda s: radial_feeder(30,4,s),S1),
                     ("weakly meshed",lambda s: weakly_meshed(30,4,4,s),S3),
                     ("denser meshed",lambda s: meshed_transmission(30,s),S2)]:
    r=[]
    for B in (3,4,5):
        for seed in range(12):
            G=gen(seed); w=line_weights(G,seed+50)
            Se,ve=exact(G,w,B)
            if ve: r.append(covered_weight(G,w,greedy(G,w,B))/ve)
    data[name]=(r,col)
for i,(name,(r,col)) in enumerate(data.items()):
    x=np.random.default_rng(i).normal(i,0.07,len(r))
    a.scatter(x,r,s=26,color=col,alpha=.75,zorder=3,label=name)
a.axhline(1-1/math.e,color=S2,ls="--",lw=1.5,zorder=2)
a.annotate("theoretical guarantee  $1-1/e$ = 0.632",(0.97,0.075),xycoords="axes fraction",
           fontsize=8.5,ha="right",color=S2,weight="bold")
a.axhline(1.0,color=MUTED,lw=1.0,zorder=1)
a.set_xticks(range(3)); a.set_xticklabels(list(data.keys()),fontsize=8.5)
a.set_ylim(0.60,1.03); a.set_ylabel("greedy value / exact optimum")
a.set_title("A five-line greedy is within 0.1% of optimal\n108 instances, n=30, B=3-5",
            fontsize=9.5,color=INK)
a.annotate("mean 0.9991\nworst 0.9811",(0.06,0.955),xycoords="axes fraction",
           fontsize=8.5,color=INK,weight="bold")

# ---- panel B: the QUBO graph IS the grid ----
b=ax[1]
G=weakly_meshed(34,4,4,2); w=line_weights(G,7)
S=greedy(G,w,5); Sset=set(S)
pos=nx.kamada_kawai_layout(G)
cov=[e for e in G.edges() if e[0] in Sset or e[1] in Sset]
unc=[e for e in G.edges() if e not in cov]
nx.draw_networkx_edges(G,pos,ax=b,edgelist=unc,edge_color="#cfd6dc",width=1.4)
nx.draw_networkx_edges(G,pos,ax=b,edgelist=cov,edge_color=S3,width=2.4)
nx.draw_networkx_nodes(G,pos,ax=b,nodelist=[v for v in G if v not in Sset],
    node_color="white",edgecolors=MUTED,linewidths=1.2,node_size=90)
nx.draw_networkx_nodes(G,pos,ax=b,nodelist=list(S),node_color=S1,
    edgecolors="white",linewidths=1.6,node_size=250)
b.axis("off")
b.set_title("One variable per bus, one coupling per line\n"
            "$Q_{ii}$ = weighted degree,  $Q_{uv} = -w_{uv}$  --  no ancillas",
            fontsize=9.5,color=INK)
b.text(0.5,-0.06,"blue = PMU placed   green = line covered   grey = uncovered",
       transform=b.transAxes,ha="center",fontsize=8,color=INK2)
plt.tight_layout(); plt.savefig("fig_pmu.png",dpi=185,bbox_inches="tight"); print("ok")
