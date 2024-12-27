import sympy
import numpy
import os
import sys
import sparse
import itertools
from pathlib import Path

# Determine the base directory (where the script is located)
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the data directory exists
J_file = DATA_DIR / "sparse_J_matrices.py"

# Sparse array conversion function
def SparsAr(dict, shape, **args):
    try:
        return sympy.ImmutableSparseNDimArray(dict, shape, **args)
    except:
        return sympy.ImmutableSparseNDimArray(
            {k: sympy.sympify(d) for k, d in dict.items()},
            tuple(sympy.sympify(k) for k in shape),
            **args
        )

# Initialize J_dict as an empty dictionary
J_dict = {}

def J_init():
    """Clears the J_dict dictionary."""
    J_dict.clear()

# Load J_dict from the sparse_J_matrices.py file if it exists
if J_file.exists():
    sys.path.append(str(DATA_DIR))  # Add the data directory to the system path
    try:
        from sparse_J_matrices import J_dict
    except ImportError:
        J_dict = {}

# Save J_dict to the sparse_J_matrices.py file
def save_J_dict():
    """ saves dictionary J_dict into file J_file, in a human readable way
    also converts sparse with "object" dtype into sympy.ImmutableSparseNDimArray
    """
    s="import sparse\nimport sympy\ndef SparsAr(dict,shape,**args):\n try: return sympy.ImmutableSparseNDimArray(dict,shape,**args)\n except: return sympy.ImmutableSparseNDimArray({k:sympy.sympify(d) for k,d in dict.items()},tuple(sympy.sympify(k) for k in shape),**args)\nJ_dict={\n"
    for M in list(J_dict):
        if isinstance(J_dict[M],sparse.COO):
            s=s+str(M)+":sparse.COO("+str(J_dict[M].coords.tolist())+","+str(J_dict[M].data.tolist())+","+str(J_dict[M].shape)+"),\n"
        else:
            s=s+str(M)+":SparsAr({"+",".join(str(k)+":"+(str(c) if sympy.sympify(c).is_integer else 'sympy.sympify("{}")'.format(c)) for k in J_dict[M]._sparse_array for c in [J_dict[M]._sparse_array[k]])+"},"+str(J_dict[M].shape)+"),\n"
    s=s+"}\n"
    with open(J_file,"w") as f:f.write(s)
    pass

##########################################
# Manipulation of vectors in tensor repr #
##########################################

class vec:
    def __init__(self,*m):
        """ eg vec(1,-1,2) is the vector |1,-1,3> """
        if len(m)==1 and type(m[0])==dict: self.terms=m[0] # allow initialisation with a dict
        elif len(m)==1 and hasattr(m[0],"__iter__"):
            self.terms={tuple(m[0]):1}
        else:
            self.terms={tuple(m):1}
    def n(self):
        """ the number of tensor products in this tensor irrep """
        if self.terms=={}: raise ValueError("no fixed dimension")
        return len(list(self.terms)[0])
    def __repr__(self):
        if all(self.terms[t]==0 for t in self.terms): return "0"
        return "".join(("+" if n!=0 and not sf.startswith("-") else "")+
                ("" if f==1 else "-" if f==-1 else ((sf if all(x not in sf[1:] for x in "+-") else ("("+sf+")"))))+
                "|"+"".join(str(k)  for k in t)+">"
                for n,t in enumerate(list(self.terms))
                       for f in [self.terms[t]] for sf in [str(f)] if f!=0)
    def __add__(A,B):
        if B==0: return A
        if A.n()!=B.n(): raise ValueError("cannot add vectors with distinct 'n'")
        d={}
        for dd in [A.terms,B.terms]:
            for tp in dd:
                d[tp]=d.get(tp,0)+dd[tp]
        return vec(d)
    def __mul__(self,f): # Multiplication of a vector and a number
        return vec({t:f*self.terms[t]
                       for t in self.terms})
    def __sub__(A,B): return A+(-1*B)
    def __neg__(A): return -1*A
    def Pact(self,sigma,inplace=True):
        r""" computes P_\sigma|self>, where P_\sigma is the Young Symmetrizer"""
        dct={
            tuple([k[i] if i+1 not in sigma else k[sigma[sorted(sigma).index(i+1)]-1]
                   for i in range(len(k))]) : self.terms[k] for k in self.terms}
        if not inplace: return vec(dct)
        self.__init__(dct); return self
    def cact(self,Lambda,inplace=True):
        r""" computes c_\Lambda|self>, where c_\Lambda is the Young Symmetrizer"""
        # See eg the paragraph after (A.44) in my PhD
        Tableau=[list(range(1+sum(Lambda[:k]),1+sum(Lambda[:k+1])))
            for k in range(len(Lambda))]
        Res=self
        for k in range(Lambda[0]): #columns
            idx=[t[k] for t in Tableau if len(t)>k]
            Res=sum((-1)**(sum(1 for i in range(len(idx)) for j in range(i+1,len(idx)) if sigma[i]>sigma[j])
                           )*Res.Pact(sigma,inplace=False)
                     for sigma in itertools.permutations(idx))
        for k in range(len(Lambda)):#lines
            idx=Tableau[k]
            Res=sum(Res.Pact(sigma,inplace=False)
                for sigma in itertools.permutations(idx))
        if inplace: 
            self.__init__(Res.terms)
            return self
        else: return Res
    def J_act(self,i,j):
        """ computes J(i,j)|self> """
        d={}
        for t in self.terms:
            for (a,b,sgn) in [(i,j,1),(-j,-i,-1)]: #two terms in J(i,j)
                for n in range(len(t)):
                    if t[n]==b:
                        tt=tuple([z if nn!=n else a for nn,z in enumerate(t)])
                        d[tt]=d.get(tt,0)+self.terms[t]*sgn
        if d=={}: d={list(self.terms)[0]:0}
        return vec(d)


    def to_col(self,bv):
        """given a list bv of basis vectors, write |self> in this basis, after adding a vector to the list if needed"""
        if all(self.terms[t]==0 for t in self.terms): return [0]*len(bv)
        tpls=set(sum((list(v.terms) for v in bv+[self]),[])) ## All the tuples that appear
        M=sympy.Matrix([[bv[i].terms.get(t,0) for i in range(len(bv))] for t in tpls])
        R=sympy.Matrix([self.terms.get(t,0) for t in tpls])#RHS
        B=sympy.Matrix([[M,R]])
        if (B.T*B).det()!=0: 
            bv.append(self)
            return [0]*(len(bv)-1)+[1]
        return list(M.solve(R))

vec.__rmul__=vec.__mul__
vec.__radd__=vec.__add__


###################################################################
# Manipulation of symmetric vectors (particular case of previous) #
###################################################################

class symvec:
    def __init__(self,*m):
        # print(m)
        if len(m)==1 and type(m[0])==dict: self.terms=m[0]
        else : self.terms={tuple(m):1} # the coefficient of each "m" (ie multiplicity) in the linear combination of them
        self.r=len(list(self.terms)[0])//2
    def s(self):
        if self.terms=={}: raise ValueError("no fixed dimension")
        return sum(list(self.terms)[0])
    def __repr__(self):
        if all(self.terms[t]==0 for t in self.terms): return "0"
        return "".join(("+" if n!=0 and not sf.startswith("-") else "")+
                ("" if f==1 else "-" if f==-1 else ((sf if all(x not in sf[1:] for x in "+-") else ("("+sf+")"))))+
                ("(" if t.count(0)<len(t)-1 else "")+
                "|"+"".join(str(k)  for k in (list(range(1,self.r+1))+list(range(-1,-self.r-1,-1))) for n in range(t[k-int(k>0)]) )+">"+
                (" + perm )" if t.count(0)<len(t)-1 else "")
                for n,t in enumerate(list(self.terms))
                       for f in [self.terms[t]] for sf in [str(f)] if f!=0)
    def __add__(A,B):
        if A.s()!=B.s(): raise ValueError("cannot add vectors with distinct 's'")
        d={}
        for dd in [A.terms,B.terms]:
            for tp in dd:
                d[tp]=d.get(tp,0)+dd[tp]
        return symvec(d)
    def __mul__(self,f):
        return symvec({t:f*self.terms[t]
                       for t in self.terms})
    def J_act(self,i,j):
        """ computes J(i,j)|self> """
        d={}
        for t in self.terms:
            for (a,b,sgn) in [(i,j,1),(-j,-i,-1)]: #two terms in J(i,j)
                if t[b-int(b>0)]>0: #it has at least one b
                    tt=tuple([t[i-int(i>0)]+int(i==a)-int(i==b)
                              for i in (list(range(1,self.r+1))+list(range(-self.r,0)))])
                    mul=tt[a-int(a>0)]
                    d[tt]=d.get(tt,0)+mul*self.terms[t]*sgn
        if d=={}: d={list(self.terms)[0]:0}
        return symvec(d)
    def to_col(self,bv):
        """given a list bv of basis vectors, write |self> in this basis, after adding a vector to the list if needed"""
        if all(self.terms[t]==0 for t in self.terms): return [0]*len(bv)
        cfs=[sympy.Symbol("c"+str(i)) for i in range(len(bv))]
        tpls=set(sum((list(v.terms) for v in bv+[self]),[])) ## All the tuples that appear
        eqns=[sum(cfs[i]*bv[i].terms.get(t,0) for i in range(len(bv)))-self.terms.get(t,0)
            for t in tpls]
        d=sympy.solve(eqns,cfs)
        if d==[]:
            bv.append(self)
            return [0]*(len(bv)-1)+[1]
        assert type(d)==dict
        return [d[c] for c in cfs]
symvec.__rmul__=symvec.__mul__


#############################################
# Function that creates matrices for J(i,j) #
#############################################
#def J_matrix(i,j,s,r,a=1,allrepr=False):
def J_matrix(*largs,**args):
    """
    computes the matrix for J(i,j) in irrep (a,s) of so(2,r)
    if allrepr is True, then do not load it from a file, and return all J-matrices plus 
    syntax: 
    J_matrix(i,j,s,r) for a symmetric repr
    J_matrix(i,j,s,r,a) for a rectangular repr
    J_matrix(i,j,Lambda,r) for a Young diag
    J_matrix(s,r,allrepr=True), J_matrix(s,r,a,allrepr=True) and J_matrix(Lambda,r,allrepr=True) are also fine
    """
    from itertools import permutations
    largs=list(largs)
    i,j,s,r,a,Lambda,allrepr=(args.get(k,None) for k in ["i","j","s","r","a","Lambda","allrepr"])
    na=len(largs)+sum(1 for k in (i,j,s,r,a,Lambda) if k!=None) # nb of arguments
    if na==4 and a==None:
        i,j,s,r=(largs.pop(0) if k==None else k for k in [i,j,(s if Lambda==None else Lambda),r])
        if hasattr(s, '__iter__'):s,Lambda=None,s
    elif na==5: i,j,s,r,a=(largs.pop(0) if k==None else k for k in [i,j,s,r,a])
    elif allrepr and na==2 and a==None:
        s,r=(largs.pop(0) if k==None else k for k in [(Lambda if Lambda else s),r])
        if hasattr(s, '__iter__'):s,Lambda=None,s
    elif allrepr and na==3: s,r,a=(largs.pop(0) if k==None else k for k in [s,r,a])
    else: raise ValueError("unable to parse arguments of function J_matrix")


    if not Lambda:
        Lambda=tuple([s]*a+[0]*(r-a))
        a,s=None,None

    
    # from IPython.display import display, Markdown
    if s and s<0: raise ValueError("These finite-dimensionnal irreps exist only for s>=0")
    if not allrepr:
        if i<0 and j<0: return -J_matrix(-j,-i,**{k:v for (k,v) in [("s",s),("r",r),("a",a),("Lambda",Lambda)] if v!=None})
        if (i*j<0) and abs(i)>abs(j): return -J_matrix(-j,-i,**{k:v for (k,v) in [("s",s),("r",r),("a",a),("Lambda",Lambda)] if v!=None})
        if i>r: return J_matrix(i-2*r-1,j,**{k:v for (k,v) in [("s",s),("r",r),("a",a),("Lambda",Lambda)] if v!=None})
        if j>r: return J_matrix(i,j-2*r-1,**{k:v for (k,v) in [("s",s),("r",r),("a",a),("Lambda",Lambda)] if v!=None})
        if i+j==0: return 0*J_matrix(1,1,**{k:v for (k,v) in [("s",s),("r",r),("a",a),("Lambda",Lambda)] if v!=None})
    else :
        oldJ_dict=J_dict.copy()
        [J_dict.pop(k) for k in list(J_dict)] #save old value of J_dict, to avoid impacting it
    tup=(i,j,s,r,a) if s!=None else (i,j,Lambda,r)
    if tup not in J_dict or allrepr:
        print("computing G repr for ",tup)
        # do the stuff here
        #######bv=[symvec(s,*([0]*(2*r-1)))] was the old version
        if a==1: bv=[vec(*([1]*s))]
        elif s==1 and a<=2*r: bv=[vec({p:(-1)**sum(1 for i in range(a) for j in range(i+1,a) if p[i]>p[j])
                            for p in permutations(
                                       range(1,a+1) if a<=r else
                                                  list(range(-r,0))+list(range(1,a+1-r)))})]
        elif Lambda!=None:
            bv=vec(sum(([k+1]*(Lambda[k]) for k in range(len(Lambda))),[]))
            bv=[bv.cact(Lambda)]
        else: raise NotImplementedError
        Js={(i,j):
            sparse.DOK((0,0),dtype="object")
            for i in range(-r,r+1) for j in range(-r,r+1) if i*j!=0 and i+j!=0 and (not(i<0 and j<0)) and (not((i*j<0) and abs(i)>abs(j)))}
        while True: #add vectors one after another and simultaneously, add columns to all matrices
            l=[I for I in Js if Js[I].shape[1]<len(bv)]
            if l==[]: break # done: all matrices already have the right size
            k,l=l[0]
            v=bv[Js[(k,l)].shape[1]] #vector the image of which was not inserted yet
            res=v.J_act(k,l)
            nc=res.to_col(bv)
            Js[(k,l)]=sparse.DOK((len(bv),Js[(k,l)].shape[1]+1),Js[(k,l)].data,dtype="object") #fill with zeroes
            for n,c in enumerate(nc):Js[(k,l)][n,-1]=c
        is_int=all(sympy.sympify(d[K]).is_integer for (k,l) in Js for d in [Js[k,l].data] for K in d)
        for (k,l) in Js:
            if is_int:   Js[k,l]=sparse.DOK(Js[k,l].shape,Js[k,l].data,dtype="int").to_coo()
            else:        Js[k,l]=SparsAr(Js[k,l].data,Js[k,l].shape)
            J_dict[(k,l)+tup[2:]]=Js[k,l]
        if allrepr:
            JJ=J_dict.copy()
            [J_dict.pop(k) for k in list(J_dict)]
            [J_dict.__setitem__(k,oldJ_dict[k]) for k in oldJ_dict]
            return [JJ,bv]
        save_J_dict()
    return J_dict[tup]