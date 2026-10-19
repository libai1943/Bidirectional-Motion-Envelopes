"""Create original vector geometry artwork and independently audit the bounds.

These checks are manuscript verification, not the paper's simulation study.
Run with Python, numpy, and matplotlib available.
"""
from pathlib import Path
import math
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def expansions(s, k, f, r, b):
    sp, sm = max(s, 0.0), max(-s, 0.0)
    ef, eb = (1+b*abs(k))*sp, (1+b*abs(k))*sm
    el = max(-r*k*sp, (f+sp/2)*k*sp) + max(-f*k*sm, (r+sm/2)*k*sm)
    er = max(r*k*sp, -(f+sp/2)*k*sp) + max(f*k*sm, -(r+sm/2)*k*sm)
    return np.array([ef, eb, el, er])

def vertices(s, k, f, r, b, mode='arc', n=101):
    w = np.linspace(0.0, s, n)
    th = k*w
    ct, st = np.cos(th), np.sin(th)
    if mode == 'arc' and k != 0:
        px = w*np.sinc(th/np.pi)
        # Stable near zero curvature.
        py = (2*np.sin(th/2)**2)/k
    else:
        px, py = w, np.zeros(n)
    corners = np.array([[f,b],[f,-b],[-r,-b],[-r,b]])
    x = px[:,None] + ct[:,None]*corners[:,0] - st[:,None]*corners[:,1]
    y = py[:,None] + st[:,None]*corners[:,0] + ct[:,None]*corners[:,1]
    return np.stack((x,y), axis=-1)

def checks():
    rng = np.random.default_rng(20261011)
    worst = {'arc':np.full(4,-np.inf), 'affine':np.full(4,-np.inf)}
    counts = {m:0 for m in worst}
    cases = []
    for _ in range(6000):
        f,r,b = rng.uniform(.15,5),rng.uniform(.15,4),rng.uniform(.1,2)
        k = rng.uniform(.001,.99)/b * rng.choice([-1,1])
        s = rng.uniform(0,np.pi/2)/abs(k) * rng.choice([-1,1])
        cases.append((s,k,f,r,b))
    for k in [-.4,-.1,-1e-10,0,1e-10,.1,.4]:
        for s in [-1,-.2,0,.2,1]:
            cases.append((s,k,3.76,.929,.971))
    for s,k,f,r,b in cases:
        ef,eb,el,er = expansions(s,k,f,r,b)
        for mode in worst:
            xy = vertices(s,k,f,r,b,mode,n=201)
            x,y=xy[:,:,0],xy[:,:,1]
            violations=np.array([np.max(x)-(f+ef),(-r-eb)-np.min(x),
                                  np.max(y)-(b+el),(-b-er)-np.min(y)])
            worst[mode] = np.maximum(worst[mode],violations)
            counts[mode] += xy.shape[0]*xy.shape[1]
            assert violations.max()<2e-9,(mode,s,k,f,r,b,violations)
        e2 = expansions(s,-k,f,r,b)
        assert np.allclose(e2, [ef,eb,er,el])
    report = {'seed':20261011, 'segments_per_model':len(cases),
              'body_vertex_evaluations':counts,
              'max_signed_bound_violation_m':{m:v.tolist() for m,v in worst.items()},
              'includes_negative_curvature':True,
              'includes_straight_and_stationary':True,
              'angular_range_rad':[0,math.pi/2],
              'scope':'Independent formula audit; not a replacement for analytical proofs or experimental results.'}
    (ROOT/'audit').mkdir(exist_ok=True)
    (ROOT/'audit'/'geometry-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

def figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
        'mathtext.fontset':'stix','font.size':9,'pdf.fonttype':42})
    blue,orange,ink='#1764a0','#c56a28','#343b42'
    f,r,b,k,ell=3.76,.929,.971,.18,.8
    fig,axs=plt.subplots(1,2,figsize=(7.16,2.38))
    fig.subplots_adjust(left=.012,right=.992,bottom=.15,top=.89,wspace=.07)
    for i,(ax,s,color) in enumerate(zip(axs,[ell,-ell],[blue,orange])):
        ef,eb,el,er=expansions(s,k,f,r,b)
        xmin,xmax,ymin,ymax=-r-eb,f+ef,-b-er,b+el
        ax.add_patch(Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,
            fc='#f0f5f8',ec=ink,lw=1.05,ls=(0,(4,2)),zorder=1))
        swept=vertices(s,k,f,r,b,n=30)
        for xy in swept[1:]:
            ax.add_patch(Polygon(xy,fc='none',ec=color,lw=.34,alpha=.38,zorder=2))
        ax.add_patch(Polygon(swept[0],fc='white',ec=ink,lw=1.15,zorder=3))
        ax.add_patch(Polygon(swept[-1],fc='none',ec=color,lw=1.2,zorder=4))
        # Body-fixed axes, with the rear axle as origin in both panels.
        ax.plot(0,0,'o',ms=2.8,color=ink,zorder=6)
        for end,label,offset in [((.7,0),'$X$',(.08,-.08)),((0,.65),'$Y$',(.13,-.04))]:
            ax.annotate('',xy=end,xytext=(0,0),arrowprops={'arrowstyle':'->','lw':.8,'color':ink},zorder=5)
            ax.text(end[0]+offset[0],end[1]+offset[1],label,fontsize=9)
        ax.text(.08,-.25,'$O$',fontsize=9)
        # Travel arrow outside the initial body keeps the frame directions unambiguous.
        start,end=(.65,2.25) if s>0 else (2.25,.65)
        ay=ymax+.24
        ax.annotate('',xy=(end,ay),xytext=(start,ay),arrowprops={'arrowstyle':'-|>','color':color,'lw':1.3})
        ax.text(1.45,ay+.08,'$s>0$' if s>0 else '$s<0$',ha='center',color=color,fontsize=9)
        def dim(a,z,label,dx=0,dy=0):
            ax.annotate('',xy=z,xytext=a,arrowprops={'arrowstyle':'|-|','lw':.7,'color':ink,'shrinkA':0,'shrinkB':0})
            ax.text((a[0]+z[0])/2+dx,(a[1]+z[1])/2+dy,label,ha='center',va='center',fontsize=9,
                bbox={'fc':'white','ec':'none','pad':.6})
        # Dimension bars separate the nominal body from the four expansions.
        if s>0:
            dim((f,ymin-.20),(xmax,ymin-.20),'$e_f$',dy=-.13)
            ax.text(xmin+.12,ymin-.32,'$e_b=0$',fontsize=9)
        else:
            dim((xmin,ymin-.20),(-r,ymin-.20),'$e_b$',dy=-.13)
            ax.text(xmax-.75,ymin-.32,'$e_f=0$',fontsize=9)
        dim((xmax+.21,b),(xmax+.21,ymax),r'$e_\ell$',dx=.22)
        dim((xmax+.21,ymin),(xmax+.21,-b),'$e_r$',dx=.22)
        ax.text(f-.38,-b+.18,'front',fontsize=8,ha='right',color=ink,zorder=6)
        ax.set_xlim(xmin-.17,xmax+.76);ax.set_ylim(ymin-.58,ymax+.62)
        ax.set_aspect('equal');ax.axis('off')
        ax.set_title('(a) Forward motion' if s>0 else '(b) Reverse motion',loc='left',fontsize=10,pad=0)
    fig.legend(handles=[Line2D([],[],color=ink,lw=1.15,label='Initial body'),
        Line2D([],[],color='#9dabb6',lw=.7,label='Intermediate bodies'),
        Line2D([],[],color=ink,ls='--',lw=1,label='Bidirectional motion envelope')],
        loc='lower center',ncol=3,frameon=False,fontsize=8.5,columnspacing=1.8,bbox_to_anchor=(.5,-.015))
    path=ROOT/'figures'/'forward-reverse-envelope';path.parent.mkdir(exist_ok=True)
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.02)
    fig.savefig(path.with_suffix('.png'),bbox_inches='tight',pad_inches=.02,dpi=240)
    plt.close(fig)

if __name__=='__main__':
    checks();figure()
