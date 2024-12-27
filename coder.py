# Import necessary libraries
import numpy, sympy, subprocess, tempfile, os, functools, itertools, random, functools, time
import sparse # For working with sparse matrices
from IPython.display import display, Markdown, HTML, Latex # For displaying formatted outputs in Jupyter Notebooks
import sparse_J_matrix  # Custom module (assumed to handle specific sparse matrix operations)


# Utility function to create sparse multidimensional arrays
def SparsAr(dict,shape,**args):
    """
    Creates a sympy ImmutableSparseNDimArray from a dictionary of values and a shape.
    Converts input values and shape to symbolic representations if necessary.
    """
    try: return sympy.ImmutableSparseNDimArray(dict,shape,**args)
    # If conversion fails, sympify values and shape before creating the array
    except: return sympy.ImmutableSparseNDimArray({k:sympy.sympify(d) for k,d in dict.items()},tuple(sympy.sympify(k) for k in shape),**args)


# Wrapping sparse matrix operations for fallback compatibility
try: sp_tensordot=sparse.tensordot
except: sp_tensordot=sparse.COO.tensordot
try: sp_transpose=sparse.transpose
except: sp_transpose=sparse.COO.transpose

# NumPy equivalents for tensor operations
np_tensordot,np_transpose=numpy.tensordot,numpy.transpose                

# Function to execute bash commands and return output
def get_output(s):
    """
    Wrapper around the bash function to execute shell commands.
    """
    return bash(s,print_input=False)

# Bash execution utility with optional output logging
def bash(s,print_input=True,print_output=False):
    """
    Executes a shell command and returns its output.
    Handles temporary files for cases where subprocess.check_output fails.
    """
    if print_input: print("$ "+s) # Log the command being executed
    try:
        # Attempt to use subprocess for command execution
        r=subprocess.check_output(["/bin/sh","-c",s])
    except:
        # Fallback: Execute the command and capture output via temporary file
        tmpd=tempfile.mkdtemp()
        ss=s+r" >{}/a.b".format(tmpd)
        os.system(ss)
        with open("{}/a.b".format(tmpd),"r") as f: r=f.read()
        os.system('rm -R '+tmpd) # Clean up temporary directory
    if print_output: print(r)
    if type(r)==bytes:
        try:
            r=r.decode("utf-8") # Decode byte output if applicable
        except: pass
    return r


# Initialize symbolic variables and functions
g,z,r=(sympy.Symbol(l) for l in "gzr")  # Define symbols g, z, and r
list_sing_ind=[g]    #symbols with a single index => useless, nothing has more than one index
list_scalars=[z,r]   #symbols with a single index => useless, nothing has more than one index
tr=sympy.Function("trace")
u,x,chigl,chiso,gzrat=(sympy.Function(k) for k in ["u","x","chigl","chiso","gzrat"])

# LaTeX rendering customization
dsl={"chigl":r"\chi^{(gl)}","chiso":r"\chi^{(so)}"}
def subscript_latex(e,p,exp=None):
    """
    Custom LaTeX formatting for symbolic expressions with subscripts and exponents.
    """
    if exp:
        tmp=subscript_latex(e,p)
        return ("{%s}"%tmp if "^" in tmp else tmp)+r"^{%s}"%sympy.latex(exp)
    else :
        n=type(e).name
        n=dsl.get(n,n)
        return (" %s_{%s}"%(n,",".join(sympy.latex(k) for k in e.args))) #now allows for multiple args
    
# Apply custom LaTeX formatting to symbolic functions
u._latex=subscript_latex
x._latex=subscript_latex
chigl._latex=subscript_latex
chiso._latex=subscript_latex


# LaTeX rendering for gzrat expressions
def gzlat(e,p,exp=None):
    """
    Custom LaTeX formatting for gzrat symbolic expressions.
    """
    if exp==None:
        kwds=[a.name for a in e.args[1:]]
        v=e.args[0]
        if "prim" in kwds: return (r"\tfrac{1}{1-%s/g}"%v) if "shift" in kwds else (r"\tfrac{%s/g}{1-%s/g}"%(v,v))
        else : return (r"\tfrac{1}{1-%s\,g}"%v) if "shift" in kwds else (r"\tfrac{%s\,g}{1-%s\,g}"%(v,v))
    else: return r"\left(%s\right)^{%s}"%(gzlat(e,p),sympy.latex(exp))
gzrat._latex=gzlat


gzrep=(lambda x:(lambda *k:gztmp(x,*k))) # function to do a expr.replace(gzrat,gzrep(x)) to substitute g with its eigenvalue x"""
def gztmp(x,v,*kwds):
    """
    Helper function for gzrep to substitute symbolic g with eigenvalue x.
    """
    kwds=[a.name for a in kwds]
    X=(1/x if "prim" in kwds else x)
    return (1 if "shift" in kwds else X*v)/(1-X*v)

def prime(k): 
    """
    Replaces symbolic g with its prime value 1/g and modifies gzrat expressions.
    """
    return k.replace(g,1/g).replace(gzrat,lambda *k:(gzrat(*([i for i in k if i.name!="prim"]+["prim"]*all(a.name!="prim" for a in k[1:]))))) if hasattr(k,"subs") else k

class formal:
    """
    A class for manipulating symbolic variables and functions in mathematical operations,
    particularly focusing on expressions involving indices. This class enables operations
    like tensor manipulations, substitutions, and LaTeX formatting for complex expressions.
    These formal expressions are supllemented with indices, they should always have as many 
    factors as the number of indices.
    """

    # General description of attributes and methods:
    # - terms(): Extracts terms if the object represents a summation.
    # - pref(): Retrieves the prefactor of the current object.
    # - facs(): Retrieves non-prefactor factors.
    # - indices(): Returns indices associated with the object.
    # - A variety of mathematical and tensor operations are implemented as class methods.

    def terms(self):
        """Returns all terms if the object is a summation, otherwise returns itself as a single-term list."""
        return self._terms_ if self.issum else [self]

    def pref(self):
        """Extracts and returns the prefactor (indices i=j=0) from the factors."""
        return sympy.Mul(*[k for (i,j,k) in self.factors if i==j==0])

    def facs(self):
        """Returns the list of factors where indices are non-zero."""
        return [(i,j,f) for (i,j,f) in self.factors if i!=0!=j]

    def indices(self):
        """
        Retrieves the indices associated with the object.
        If the object represents a summation, returns unique index pairs from all terms.
        """
        if self.issum:
            return list(set(tuple([(a,b) for (a,b,f) in tmp.factors if a!=0!=b])
                       for tmp in self.terms()))
        else:
            return list([(a,b) for (a,b,f) in self.factors if a!=0!=b])
    
    def __init__(self,expr=None,nb_inds=None,**args):
        """
        Initializes a `formal` object.
        
        Args:
            expr: Expression to initialize. If None, initialization depends on other arguments.
            nb_inds: Number of indices for the object.
            args: Additional arguments such as 'terms' or 'factors'.

        If `expr` is None:
            - Uses `terms` from `args` for summation initialization.
            - Uses `factors` from `args` for factor initialization.
        """
        if expr==None:
          if "terms" in args:
            self._terms_=args["terms"]
            assert all(type(x)==formal or x==0 for x in self._terms_)
            self._terms_=[k for k in self._terms_ if type(k)==formal]
            if any(i.issum for i in self._terms_):
                self._terms_=[t for i in self._terms_ for t in i.terms()]
            self.issum=True # A sum does not have an explicit `nb_inds`
            return None
          if "factors" in args:
            # Handles initialization with factors
            if any( (type(t)==type(sympy.sympify(0)) if hasattr(t,"subs") else t==0)
            for (i,j,t) in args["factors"]):
                self._terms_=[]
                self.issum=True
                return None
            self.issum=False
            self.factors=args["factors"]

            # Validate indices and ensure no duplication
            II,JJ=([abs(f[k]) for f in self.factors for k in [0,1] if f[k] and ((f[k]>=0) == (k==l))] for l in [0,1])
            assert max(II)==max(JJ)==len(II)==len(JJ)

            # Set number of indices
            if nb_inds==None:
                nb_inds=max(abs(k) for (i,j,f) in self.factors for k in (i,j))
            for i in range(max(II)+1,nb_inds+1):
                self.factors.append((i,i,1))
            assert not any (I.count(x)>1 for I in [II,JJ] for x in I)
            self.nb_inds=nb_inds
            assert all(max(i,j)<=nb_inds for (i,j,e) in self.factors)
            return None
          raise NotImplementedError
        if nb_inds==1 or nb_inds==None:
             # Initialize with a single index
            nb_inds=1
            if ((type(expr)==type(sympy.sympify(0)) if hasattr(expr,"subs") else expr==0)):
                self._terms_=[]
                self.issum=True
                return None
            self.factors=[(1,1,expr)] # (1,1) denotes connection to the first pair of indices
        elif nb_inds==0:
            self.factors=[(0,0,expr)] # (0,0) denotes a prefactor
        else: raise NotImplementedError
        self.nb_inds=nb_inds
        self.issum=False
    
    def setinds(self,nbi): #set the number of indices to nbi
        if self.nb_inds==nbi: return self
        if self.issum:
            return formal(terms=[k.setinds(nbi) for k in self._terms_],nb_inds=nbi)
        else:
            return formal(factors=self.factors,nb_inds=nbi)
    
    def replace(self,*args):
        if self.issum:
            return formal(terms=[k.replace(*args) for k in self._terms_])
        else:
            return formal(factors=[(i,j,(f.replace(*args) if isinstance(f,sympy.Expr) else f)) for (i,j,f) in self.factors])
        
    def __add__(self,B):
        """
        Overloads the '+' operator for summation of `formal` objects.
        
        Args:
            B: Another `formal` object or a compatible input to initialize as `formal`.
        
        Returns:
            A new `formal` object representing the summation.
        """
        if self.issum and len(self._terms_)==0:return B # Return B if the current object has no terms
        if not isinstance(B,formal):
            B=formal(B) # Convert B to a `formal` object if it's not
        if B.issum and len(B._terms_)==0:return self # Return self if B has no terms
        # Determine the maximum number of indices for the summation
        nbi=max(k._terms_[0].nb_inds if k.issum else k.nb_inds
                for k in [self,B])
        # Create a new formal object with updated indices
        return formal(terms=[k.setinds(nbi) for t in [self,B] for k in (t.terms())],nb_inds=nbi)
    
    def tensor(self,B):
        """
        Computes the tensor product of two `formal` objects.
        
        Args:
            B: Another `formal` object or a compatible input.
        
        Returns:
            A new `formal` object representing the tensor product.
        """
        if not isinstance(B,formal):
            B=formal(B) # Convert B to a `formal` object if it's not
        if any(t.issum for t in [self,B]):
            # Handle cases where either object is a summation
            return formal(terms=[a.tensor(b)
                                 for a in self.terms() for b in B.terms()])
        # Construct factors for the tensor product
        return formal(factors=[(0,0,e)
                                for e in [sympy.Mul(*[k for (i,j,k) in self.factors+B.factors
                                                if i==0 and j==0])]
                               if e!=1
                                ]+[((i+s if i>0 else i-s),(j+s if j>0 else j-s),k) for (l,s) in [(self,0),(B,self.nb_inds)]
                                   for (i,j,k) in l.factors if i!=0!=j],
                      nb_inds=self.nb_inds+B.nb_inds)
    
    def transpose(self,I,J,inplace=False):
        """
        Transposes the indices of the `formal` object.
        
        Args:
            I: Tuple representing the new positions of input indices.
            J: Tuple representing the new positions of output indices.
            inplace: If True, modifies the current object. Otherwise, returns a new object.
        
        Returns:
            The transposed `formal` object.
        Ex:
            A.transpose((2,1,-3),(1,2,-3)) means we change indices i1,i2,i3, j1,j2,j3 into i2,i1,-j3 j1,j2,-i3
        """
        if self.issum:
            if inplace:
                [t.transpose(I,J,inplace=True) for t in self._terms_]
                return self
            else:
                return formal(terms=[t.transpose(I,J) for t in self._terms_])
        N=self.nb_inds
        # Dictionaries to remap indices based on I and J
        Idct,Jdct=({k:1+A.index(k) if k in A
              else (-1-B.index(-k) if -k in B else k)
                    for k in range(1,N+1)} for A,B in [[I,J],[J,I]])
        Idct[0]=0; Jdct[0]=0
        for i in range(1,N+1):
            Jdct[-i]=-Idct[i]
            Idct[-i]=-Jdct[i]
        if inplace:
            self.factors=[(Idct[a],Jdct[b],f)
                      for (a,b,f) in self.factors]
            return self
        else :
            return formal(factors=[(Idct[a],Jdct[b],f)
                      for (a,b,f) in self.factors])
    
    def mprime(self,Prime,inplace=False):
        """
        Applies a 'prime' operation to specified indices.
        
        Args:
            Prime: List of indices to apply the operation.
            inplace: If True, modifies the current object. Otherwise, returns a new object.
        
        Returns:
            A new or updated `formal` object with the prime operation applied.
        Ex:
            A.mprime([1,3]) should return (1-prime_1)(1-prime_3)(A)
        """
        if not inplace: self=self.copy()
        if Prime==[]: return self
        if self.issum:
            formal.__init__(self,terms=[t.mprime(Prime) for t in self._terms_])
            return self
        Prime2=list(Prime)
        for p in Prime:
            l=[n for n,(a,b,f) in enumerate(self.factors) if abs(a)==abs(b)==p]
            if len(l)==1:
                self.factors[l[0]]=next((a,b,f-prime(f)) for (a,b,f) in [self.factors[l[0]]])
                Prime2.remove(p)
        ## the primes that where vertical bars are now dealt with
        N=self.nb_inds
        del self.nb_inds
        F=self.factors
        del self.factors
        formal.__init__(self,terms=[
            (-1)**s*formal(nb_inds=N,factors=[
                (-a if a in pri else a,
                 -b if b in pri else b,
                 f) for (a,b,f) in F])
            for s in range(len(Prime2)+1)
            for pri in itertools.combinations(Prime2,s)])
        return self
    
    def __mul__(self,B):
        """
        Overloads the '*' operator for multiplication of `formal` objects.
        
        Args:
            B: Another `formal` object or a compatible input.
        
        Returns:
            A new `formal` object representing the product.
        """
        if self.issum:
            return formal(terms=[k*B for k in self.terms()])
        if not isinstance(B,formal):
            return formal(factors=[(0,0,e)
                                for e in [B*sympy.Mul(*[k for (i,j,k) in self.factors
                                                if i==0 and j==0])]
                               if e!=1
                                ]+[(i,j,k)
                                   for (i,j,k) in self.factors if i!=0!=j],
                      nb_inds=self.nb_inds)
        # if isinstance(B,formal):
        if B.issum:
            return formal(terms=[self*k for k in B.terms()])
        # now both self and B are formal expressions without sums
        assert self.nb_inds==B.nb_inds
        N=B.nb_inds
        # Tensor multiplication with contraction
        A=self.tensor(B).transpose([i+n+1 for i in range(N) for n in [0,N]],
                                   [j+n+1 for j in range(N) for n in [0,N]])
        for i in range(N):
            A.contract(i+1,i+2,inplace=True)
        return A
    
    def __neg__(self):
        return (-1)*self
    
    def __sub__(A,B): return A+(-1)*B
    
    def __rsub__(B,A): return A+(-1)*B
    
    def __repr__(self):
        """
        String representation of the `formal` object.
        
        Returns:
            A string summarizing the object.
        """
        if self.issum: return "formal(terms=["+",".join(str(i) for i in self.terms())+"])"
        else: return "formal(factors=["+",".join(str(i) for i in self.factors)+"], nb_inds="+str(self.nb_inds)+")"

    def sort(self):
        """
        Standardizes the order of factors in the `formal` object.

        For summation objects (`issum=True`), each term is individually sorted,
        and duplicate or zeroed terms are removed. For single objects, factors are rearranged
        based on a standard order that depends on indices and prefactors.

        Returns:
            The sorted `formal` object.
        """
        if self.issum:
            # Iterate through terms in reverse to identify duplicates and simplify terms
            for n in range(len(self._terms_))[::-1]:
                self._terms_[n].sort()
                if any (hash(c)==hash(0) and c==0 for (a,b,c) in self._terms_[n].factors):
                    self._terms_.pop(n)
#           # Sort terms based on index order
            self._terms_.sort(key=lambda i:[sum(1 for (a,b,c) in i.factors if a<0)]+[(a,b) for (a,b,c) in i.factors if a!=0])
            return self
        else:
            # Standardize factor ordering for a single object
            fctrs=[(i,j,k) if (i>=0 and j>=0) or ((i*j)<0 and abs(i)<abs(j)) else (-j,-i,prime(k))  for (i,j,k) in self.factors]
            fctrs.sort(key=(lambda x:(abs(x[0]),x[0],x[1])))
            self.factors=fctrs
            return self

    def simplify(self,inplace=True):
        """
        Simplifies the `formal` object by combining similar terms and removing redundant ones.

        Args:
            inplace: If True, modifies the object directly. Otherwise, returns a new simplified object.

        Returns:
            The simplified `formal` object.
        """
        if not inplace:
            return self.copy().simplify()
        self.sort()
        N=len(self.terms())-1
        for i in range(N)[::-1]:
            if self._terms_[i].indices()!=self._terms_[N].indices():
                N=i
                continue
            else:
                # we have the same lines as the i+1, i+2 ... up to N
                ident=[j for j in range(i+1,N+1) if all(l in self._terms_[j].factors for l in self._terms_[i].factors if l[0]!=0)]
                if ident:
                    j,=ident
                    new_pref=(self._terms_[j].pref()+self._terms_[i].pref())
                    if new_pref==0:
                        self._terms_.pop(j)
                        self._terms_.pop(i)
                        N=N-2
                        continue
                    self._terms_[i]=(new_pref/self._terms_[i].pref())*self._terms_[i]
                    self._terms_.pop(j)
                    N=N-1
        return self

    def formal_sum(self):
        """
        Converts the `formal` object into a summation of its terms.

        Returns:
            A `formal` object representing the sum of all terms.
        """
        if self.issum:
            return sum([k for k in [t.formal_sum() for t in self._terms_]])
        else:
            pref=self.pref()
            nb_max=len(self.indices())
            self.factors=[(a,b,f) for (a,b,f) in self.factors if a!=0!=b]
            exp=[(a,b,f) for (a,b,f) in self.factors 
                if type(f)==sympy.Add or (type(f)==sympy.Mul and type(f.expand())==sympy.Add)]
            others=[(a,b,f) for (a,b,f) in self.factors if (a,b,f) not in exp]
            exp=[(a,b,k) for (a,b,f) in exp for k in f.expand().args]
            if len(exp)==0:
                self.factors=[(0,0,pref)]+others
                return self
            if len(others)==0:
                return self
            tmp=[]
            for k in range(len(exp)):
                if sympy.simplify(exp[k][-1].as_ordered_factors()[0]).is_number:
                    extr_pref=exp[k][-1].as_ordered_factors()[0]
                    new_exp_k=[(exp[k][0],exp[k][1],sympy.Mul(*[k for k in exp[k][-1].as_ordered_factors()[1:]]))]
                    self.factors=[(0,0,extr_pref*pref)]+others+new_exp_k
                else:
                    self.factors=[(0,0,pref)]+others+[exp[k]]
                tmp.append(self.factors)
        return sum([formal(factors=k,nb_inds=nb_max) for k in tmp])

    def gzrat_change(self):
        """
        Updates gzrat factors in the `formal` object by applying predefined transformations.

        Returns:
            The updated `formal` object with gzrat factors adjusted.
        """
        if self.issum:
            return sum([t.gzrat_change() for t in self._terms_])
        pref=self.pref()
        if type(pref)==sympy.Mul:
            pref=sympy.Mul(*[n for n in [k if type(k)!=tr
                                else (tr(gzrat(k.args[0].args[0])) if k.args[0] in [gzrat(k.args[0].args[0]),gzrat(k.args[0].args[0],"prim")]
                                else (tr(1)+tr(gzrat(k.args[0].args[0])) if k.args[0] in [gzrat(k.args[0].args[0],"shift"),gzrat(k.args[0].args[0],"shift","prim"),gzrat(k.args[0].args[0],"prim","shift")] else None))
                                        for k in pref.args]
                            ])
        elif type(pref)==sympy.Pow:
            pow,exp=pref.args[1],pref.args[0]
            pref=pref if type(exp)!=tr else (sympy.Mul(*[n for n in [k if type(k)!=gzrat else (tr(gzrat(k.args[0])) if k in [gzrat(k.args[0]),gzrat(k.args[0],"prim")]
                                                        else (tr(gzrat(k.args[0]))+tr(1) if k in [gzrat(k.args[0],"shift"),gzrat(k.args[0],"shift","prim"),gzrat(k.args[0],"prim","shift")] else None)) 
                                                                      for k in exp.args]]))**pow
        self.factors=[(a,b,f) for (a,b,f) in self.factors if a!=0!=b]
        tmp=[f for (a,b,f) in self.factors]
        if any(type(f)==sympy.Pow and type(f.args[0])==gzrat for f in tmp):
            self.factors=([(a,b,f if type(f) not in [gzrat,sympy.Pow] or f.args[0]==gzrat(f.args[0].args[0]) or f.args[0]==gzrat(f.args[0].args[0],"prim")
                                else ((gzrat(f.args[0].args[0])+1)**(f.args[1]) if str(f.args[0].args[1])=="shift" 
                                and len(f.args[0].args)==2 else (gzrat(f.args[0].args[0],"prim")+1)**(f.args[1]))
                    ) for (a,b,f) in self.factors])
        else:
            self.factors=([(a,b,f if type(f)!=gzrat or f==gzrat(f.args[0]) or f==gzrat(f.args[0],"prim")
                                else (gzrat(f.args[0])+1 if str(f.args[1])=="shift" 
                                and len(f.args)==2 else (gzrat(f.args[0],"prim")+1))
                    ) for (a,b,f) in self.factors])
        return (pref*self)

    def latex(self):
        """
        Generates a LaTeX representation of the `formal` object.

        This method constructs a detailed LaTeX string for visualizing the structure of the `formal` object,
        including its terms and indices. If the object represents a summation, all terms are combined.
        Special formatting is applied to objects with indices.

        Returns:
            str: A LaTeX representation of the object.
        """
        terms=self.terms()
        if len(terms)==0: return "0"
        elif terms[0].nb_inds==0:
            assert all(t.nb_inds==0 for t in terms)
            return sympy.latex(sum(t.pref() for t in terms))
        return "".join([("+" if n>=1 and not lpref.startswith("-") else "")+("" if pref==1 else ("-" if pref==-1 else lpref))+r"\,\begin{tikzpicture}[baseline=(n1.base)]"+"\n".join([""]+[
            r"\path %s node[%s] (n%s) {%s};"
            %(("" if l==0 else "(n%s.east)"%(l)),
              ",".join(["right=.15cm"]*(l>0)+["inner sep=0pt,outer sep=0pt" if lf[l]=="" else "draw"]),
              l+1,lf[l]) for l in range(len(lf))]+[
                  r"\path (n%s.%s) ++ (0,%s.3) node[fill,circle,inner sep=.05cm] (%s%s){};"%(l,d,s,ij,l) for l in range(1,1+len(lf)) for (s,d,ij) in [("-","south","j"),("","north","i")]]+[
                      r"\draw[green!50!black] "+" ".join([("(i%s) -- (n%s.north)"%(li,l)) if li>0 else ("(n%s.north).. controls ++(-.15,.3) .. (j%s) "%(l,-li)) for l in range(1,1+len(lf)) for li in [flf[l-1][0]]])+";"]+[
                          r"\draw[blue] "+" ".join([("(j%s) -- (n%s.south)"%(j,l))
                                              if j>0 else
                                              ("(n%s.south).. controls ++(.15,-.3) .. (i%s) "%(l,-j))
                                              for l in range(1,1+len(lf)) for j in [flf[l-1][1]]])+";"]+[""])+r"\end{tikzpicture}\,"
            for n,f in enumerate(terms)
            for AA in [any(K!=1 for I,J,K in f.factors if I*J!=0)]
            for pref in [f.pref()] for lpref in [(r"\left({}\right)" if type(pref)==sympy.Add else "{}").format(sympy.latex(pref))]
            for flf0 in [[(i,j,k) for (i,j,k) in f.factors if i!=0]]
            for flf in [flf0+[(i,i,1) for i in range(len(flf0)+1,f.nb_inds+1)]]
            for lf in [ [(r"\ensuremath{%s}"%(sympy.latex(k)) if AA else "") for (i,j,k) in flf]]]
        )
    
    def print_nice(self):#,tikz_per_line_max=20):
        ## todo: just insert the images one after another
        tex=r"""\documentclass{standalone}

\usepackage{pdfpages,amsmath}
\usepackage{tikz}
\pagestyle{empty}
\begin{document}
$"""+(self.latex() if isinstance(self,formal) else self)+r"$\end{document}"
        tmpd=tempfile.mkdtemp()
        with open(tmpd+"/TmpSourceForImgs.tex","w") as f: f.write(tex)
        # A=bash("cd "+tmpd+";pdflatex TmpSourceForImgs.tex")
        get_output("cd "+tmpd+";pdflatex TmpSourceForImgs.tex")
        get_output("cd "+tmpd+";pdflatex TmpSourceForImgs.tex")
        get_output("cd %s;pdfcrop TmpSourceForImgs.pdf"%tmpd)
        get_output("cd "+tmpd+";convert -quality 100 -density 100x100 TmpSourceForImgs-crop.pdf  TmpImgsForMarkdown-%d.png")
        #rest=[i.split(e)[-1] for i in l.split(b)] #the things between the tikz-pictures
        # display(
        TmpM=HTML(r"""<style>.container{width:auto!important;}
 .output_area img{width:auto!important;}
 .rendered_html img{width:auto!important;}</style>"""+
                  "".join(# "$".join([""]+[
            # "   "+rest[i//2]
            # if i%2==0
            # else
            '<nobr /><img src="data:image/png;base64,%s" style="max-width:none;"/><nobr /> '%get_output("cd "+tmpd+";base64 TmpImgsForMarkdown-"+str(i# //2
            )+".png").replace("\n","")

            for i in [0]))#range(len(ts))))
            #2*len(tp)+1)]+[""]))#)
        display(TmpM)
        get_output('rm -R '+tmpd)

    def copy(self):
        """
        Creates a deep copy of the `formal` object.

        Returns:
            A new `formal` object with the same properties as the current one.
        """
        if self.issum:
            return formal(terms=[i.copy() for i in self._terms_])
        else: return formal(factors=self.factors.copy(),nb_inds=self.nb_inds)

    def contract(self,j,i,inplace=False):
        """
        Contracts two indices `j` and `i` within the `formal` object.

        Args:
            j: The first index to contract.
            i: The second index to contract.
            inplace: If True, modifies the object directly. Otherwise, returns a new object.

        Returns:
            The `formal` object with the specified indices contracted.
        
        Notes:
            - The operation does not support contracting `i` with itself or `j` with itself.
            - If the object represents a summation, each term is processed individually.
        """
        if min(i,j)<=0: raise NotImplementedError
        if self.issum:
            if inplace:
                for t in self._terms_: t.contract(j,i,inplace=True)
                return self
            else:
                return formal(terms=[t.contract(j,i,inplace=False) for t in self._terms_])
        else:
            if not inplace:self=self.copy()
            l=[(a,b,f) if a==i or b==j else (-b,-a,prime(f))
               for (a,b,f) in self.factors
               if a in [i,-j] or b in [j,-i]]
            nl=[(  a-bool(a>i) if a>=0 else a+bool(a<-j)
                ,  b-bool(b>j) if b>=0 else b+bool(b<-i)
                ,f) 
                for (a,b,f) in self.factors if a not in [0,i,-j] and b not in [0,j,-i]]#relabelling of the other factors
            pref=self.pref()
            if len(l)==1: #a line that gives a trace
                self.factors=[(0,0,pref*tr(l[0][-1]))]+nl
            elif len(l)==2:
                self.factors=([] if pref==1 else [(0,0,pref)])+nl+[
                    (next(((z-bool(z>i)) if z>0 else (z+bool(z<-j)))
                          for z in [l[0][0] if l[0][1]==j else l[1][0]]),
                     next(z-bool(z>j) if z>0 else z+bool(z<-i)
                          for z in [l[0][1] if l[0][0]==i else l[1][1]]),
                     sympy.Mul(*[f for a,b,f in l]))]
            else: raise ValueError("misunderstood contraction")
            self.nb_inds-=1
            return self
    
    def insert(self,i,e=1,inplace=False):
        """
        Inserts an extra leg (index) at position `i` with a specified factor.

        Args:
            i: The position to insert the new index.
            e: The factor associated with the new index (default is 1).
            inplace: If True, modifies the object directly. Otherwise, returns a new object.

        Returns:
            The `formal` object with the new index inserted.
        """
        if not inplace: return self.copy().insert(i,e,inplace=True)
        if self.issum:
            for t in self._terms_: t.insert(i,e,inplace=True)
        else:
            l=[(a+sympy.sign(a)*bool(abs(a)>=i),b+sympy.sign(b)*bool(abs(b)>=i),f) for (a,b,f) in self.factors]+[(i,i,e)]
            self.factors=l
            self.nb_inds=self.nb_inds+1
        return self
    
    def coefs(self,v,p):
        """
        Computes the coefficients of a polynomial expansion in terms of `v`.

        Args:
            v: The variable in the polynomial.
            p: The maximum power of `v`.

        Returns:
            A list of coefficients `[coef of v^0, coef of v^1, ..., coef of v^p]`.
        """
        if self.issum:
            l=[t.coefs(v,p) for t in self._terms_]
            assert all(type(x)==formal or x==0 for t in l for x in t)
            return [formal(terms=[t[k] for t in l if type(t[k])==formal]) if any(type(t[k])==formal for t in l) else 0
                    for k in range(p+1)]
        ### Now if we have factors:
        lf=[(a,b,coefs(f,v,p)) for (a,b,f) in self.factors]
        tmp=[
            formal(terms=[
                formal(factors=[(a,b,c[ds.count(i)]) for i,(a,b,c) in enumerate(lf)],nb_inds=self.nb_inds)
                for dis in itertools.combinations(range(len(lf)+s-1),s)
                for ds in [[b-a for (a,b) in enumerate(dis)]]])
            for s in range(p+1)
        ]
        for i in range(p+1):
            tmp[i]._terms_=[t for t in tmp[i].terms() if all(hash(f)!=hash(0) or f!=0 for (a,b,f) in t.factors)]
            if len(tmp[i].terms())<=1:
                tmp[i]= 0 if len(tmp[i].terms())==0 else tmp[i].terms()[0]
        return tmp
    # substitutions for random checks of equations
    _randreps={},{} ## careful: to have this one up to date, a formal._randreps is not enough (after an from coder import formal), one need to do a coder.formal._randreps
    
    def randinit():
        """
        Initializes or resets the random substitution cache for `formal` objects.
        """
        formal._randreps={},{}

    def randsubs(self,**args):
        """
        in-place substitution to check equality for random values of parameters
        even without random, it should replace trace(...) and chigl(...)
        args include 
        "dtype" (float128)
        "randinit" (False)
        "g" : eg set it to x(3) to substitute g and gzrat into this specific ev
        inplace (False)
        keep : eg keep=[c_0,c_1] to forbid replacing c_0 and c_1 with random values
        """
        # print("randsubs("+str(self)+",".join([""]+[k+"="+str(args[k]) for k in args]))
        if args.get("randinit",False):
            formal.randinit()
            args.pop("randinit")
        if isinstance(self,formal):
            if not args.get("inplace",False): self=self.copy()
            if self.issum:
                for t in self._terms_: t.randsubs(**args)
            else: self.factors=[(i,j,formal.randsubs(f,**args)) for (i,j,f) in self.factors]
            return self
        if isinstance(self,tensor):
            if args.get("inplace",False):
                args["inplace"]=False
                self.array=formal.randsubs(self.array,**args)
            else: return tensor(formal.randsubs(self.array,**args))
        if isinstance(self,sparse.COO):
            return numpy.vectorize(lambda i:formal.randsubs(i,**args))(self.data)
            if tensor.SympyDefault: self=sympy.ImmutableSparseNDimArray(self.todense())
        if isinstance(self,sparse.DOK):
            # if not args.get("inplace",False):
            #     self=sparse.DOK(shape=self.shape,data=self.data.copy())
            for k in self.data:
                self.data[k]=formal.randsubs(self.data[k],**args)
            if any(type(k)==numpy.float128 for k in self.data):
                self.dtype=numpy.float128
                self.fill_value=numpy.float128(self.fill_value)
            if tensor.SympyDefault: self=sympy.ImmutableSparseNDimArray(self.todense())
            return self
        if isinstance(self,sympy.ImmutableSparseNDimArray):
            # tmp={k:formal.randsubs(kk,**args)
            #                                        for k,kk in self._sparse_array.items()}
            return SparsAr({k:formal.randsubs(kk,**args)
                                                   for k,kk in self._sparse_array.items()},
                                                  self.shape)
        if hasattr(self,"real") or hasattr(self,"ceiling"): return self #self is a number of some kind
        
        assert hasattr(self,"free_symbols") #check that it is a sympy expression
        dtype=args.get("dtype",{})
        if not isinstance(dtype,dict): dtype={"All":dtype}
        dct=formal._randreps[1 if any(dtype[t]=="Q" for t in dtype) else 0]
        #FS=self.free_symbols
        FS=list(self.atoms(sympy.Symbol,sympy.Function))
        K=[i for i in FS if type(i)==tr]
        for k in K:
            if k in args.get("tmptr",{}):
                res=args["tmptr"][k]
            else:
                brgs=args.copy()
                if "dtype" in brgs: brgs["random"]=brgs.pop("dtype")
                else: brgs["random"]=brgs.get("random",True)
                res=formal(factors=[(1,1,k.args[0])],nb_inds=1).to_tensor(**brgs).trace()
                if "tmptr" in args:
                    args["tmptr"][k]=res
            self=self.replace(k,res)
            FS=list(self.atoms(sympy.Symbol,sympy.Function))
        K=[i for i in FS if type(i) in [chigl, chiso]]
        for k in K:
            if k in args.get("tmptr",{}):
                res=args["tmptr"][k]
            else:
                if type(k)==chigl:
                    assert len(k.args)==1
                    sc=k.args[0]
                    rr=args["r"]
                    res=sum(sympy.Mul(*[x(k)**(p[k-1]-p[-k]) for k in range(1,rr+1)])
                            for p in itertools.product(*([range(sc+1)]*(2*rr)))
                            if sum(p)==sc
                            )
                else:
                    aa=k.args[0] if len(k.args)==2 else 1
                    rr=args["r"]
                    ss=k.args[-1]
                    assert ss==1
                    res=((-1)**aa*sympy.Mul(*[(1-z*x(k))*(1-z/x(k)) for k in range(1,1+rr)]
                    )).series(z,0,aa+1).coeff(z**aa)
                if "tmptr" in args:
                    args["tmptr"][k]=res
            self=self.replace(k,res)
            FS=list(self.atoms(sympy.Symbol,sympy.Function))
        if "g" in args and (g in FS or any(type(f)==gzrat for f in FS)):
            self=self.replace(g,args["g"]).replace(gzrat,gzrep(args["g"]))
            FS=list(self.atoms(sympy.Symbol,sympy.Function))
        if "r" in args and (r in FS):
            self=self.replace(r,args["r"])
            FS=list(self.atoms(sympy.Symbol,sympy.Function))
        if args.get("random",True)==False:
            return self #don't do any additionnal substitution if random=False

        # remove the "keep" from FS
        FS=[k for k in FS if k not in args.get("keep",[])]

        
        fl128=False #force into numpy if any float128 is present
        rat=False #force into sympy if any rational is present
        for f in FS:
            if f in dct:
                if (not fl128) and type(dct[f])==numpy.float128: fl128=True
                if (not rat) and hasattr(dct[f],"subs"): #type(FS[f])==sympy.numbers.Rational:
                    rat=True
                continue
            dt=dtype.get(f,dtype.get("All","float128"))
            if dt.startswith("int"): rd=random.randint(10,500)
            elif dt=="Q":
                rat=True
                rd=sympy.QQ(random.random())
            elif dt.startswith("float"):
                fl128=True
                rd=numpy.float128(random.random())
            dct[f]=rd
        if rat and fl128: raise ValueError("Random substitution should not mix floating point and rational numbers")
        if len(FS):
            if fl128: expr=sympy.lambdify(FS,self)(*[dct[f] for f in FS])
            else: expr=self.xreplace(dct)
        else : expr=self
        return expr
    
    def matrix_coef(self,I,J,r,random=False):
        """ given multi-indices I and J, returns the corresponding matrix coefficient """
        cI,cJ=([K.count(i)-K.count(2*r-1-i) for i in range(r)] for K in [I,J])#sectors of I and J
        if cI!=cJ: return 0
        res=0
        rdd=(({} if random else {"random":False}) if type(random)==bool else {"dtype":random})
        for t in self.terms():
            tmp=t.pref()
            tmp=formal.randsubs(tmp,r=r,**rdd)
            for (i,j,f) in t.facs():
                indI,indJ=(I[i-1] if i>0 else 2*r-1-J[(-i)-1]),(J[j-1] if j>0 else 2*r-1-I[(-j)-1])
                if indI!=indJ:
                    break

                xg=sympy.Function("x")(indI+1-r) if indI>=r else 1/sympy.Function("x")(r-indI)
                tmp*=formal.randsubs(f,g=xg,r=r,**rdd)
            else: res+=tmp
        return res
    def to_tensor(*k,**a): return tensor(*k,**a)
formal.__radd__=formal.__add__
formal.__rmul__=formal.__mul__ #if one of them is not a "formal", then they commute


def DW(**args):
    r""" computes D^{\otimes N} w=\sum_{\sigma}\cdot
    arguments (and default) include
    term : (superseeds all other arguments) eg ["D","1","trD"] to say what to tensor and/or to contract
    N (1) : Chain length
    prime(True) : prime=True means we tensor product D-D'. (N=3,prime=[1,3]) means (D-D')⊗D⊗(D-D')
    insert (True) : if set to False, then a trD will not insert any physical space any more
    pow (1): computes D^{\otimes N} w^{pow}

-- Remark: can be turned into a derivative of g⊗g⊗g⊗g…g⊗g by setting arg gtens=p (if p factors of g are tensored this way)

    """
    if "term" in args:
        term=args["term"]
        degdct={"1":0,"D":1,"D^2":2,"D^2'":2,"trD":1,"(trD)^2":2,"tr(D^2)":2,"Dgl":1,"DtrD":2}
        N=sum(degdct[t] for t in term)
        tmp=DW(N=N,prime=[s+sum(degdct[b] for b in term[:i]) for i,t in enumerate(term) for s in (range(1,3) if t in ["D^2'","D^2","tr(D^2)","DtrD"] else ([1] if t in ["D"] else []))],**{k:args[k] for k in ["gtens","pow"] if k in args}) if N!=0 else (formal(1,nb_inds=0) if not "gtens" in args else formal(factors=[(k,k,g) for k in range(1,args["gtens"]+1)],nb_inds=args["gtens"]))
        for i,t in enumerate(term):
            if t=="1":tmp.insert(i+1,inplace=True)
            elif t in ["D","Dgl"]:pass
            elif t=="D^2": tmp.contract(i+1,i+2,inplace=True)
            elif t=="D^2'": tmp.contract(i+1,i+2,inplace=True).transpose(*([list(range(1,i+1))+[-i-1]]*2),inplace=True)
            elif t=="DtrD":
                tmp.contract(i+2,i+2,inplace=True)
            elif t=="trD":
                tmp.contract(i+1,i+1,inplace=True)
                if args.get("insert",True): tmp.insert(i+1,inplace=True)
            elif t=="(trD)^2":
                tmp.contract(i+1,i+1,inplace=True).contract(i+1,i+1,inplace=True)
                if args.get("insert",True): tmp.insert(i+1,inplace=True)
            elif t=="tr(D^2)":
                tmp.contract(i+1,i+2,inplace=True).contract(i+1,i+1,inplace=True)
                if args.get("insert",True): tmp.insert(i+1,inplace=True)
            
        return tmp
    N=args.get("N",2)
    Prime=args.get("prime",True)
    if not hasattr(Prime,"__iter__"): Prime=range(1,N+1) if Prime else []
    if "gtens" in args:
        numg=args["gtens"]
        l=[]
        for part in itertools.product(*([range(numg)]*N)): # partition of [1..N] into numg subsets for each
            part=part+tuple(range(numg)) #the last guys and the g's
            for tg in range(numg): #
                nt=part.count(tg)
                cycl=formal(factors=[(1,nt,g)]+[(k+1,k,1) for k in range(1,nt)],nb_inds=nt)
                if tg==0: tens=cycl
                else:
                    no=tens.nb_inds
                    Part=[k for k in part if k<=tg]
                    tens=tens.tensor(cycl).transpose(*([
                        [ no+1+Part[:i].count(tg)
                            if Part[i]==tg else
                          1+i-Part[:i].count(tg)
                          for i in range(len(Part))]
                ]*2))
            tens.mprime(Prime,inplace=True)
            l.append(tens)
        return formal(terms=l)
    if N==0: return 1
    if N==1: return args.get("pow",1)*(formal(gzrat(z)-prime(gzrat(z)),nb_inds=1) if (1 in Prime) else formal(gzrat(z),nb_inds=1))
    return formal(terms=[args.get("pow",1)**nbcycles(sigma)*
        formal(
            factors=[(0,0,-1)]*(p.count(True)%2)+[(-i if p[i-1] else i,-sigma[i-1] if p[sigma[i-1]-1] else sigma[i-1],(gzrat(z) if sigma[i-1]>=i else gzrat(z,"shift")))
                                                  if sigma[i-1]!=i or not (i in Prime)
                                                  else (i,i,gzrat(z)-gzrat(z,"prim"))
                for i in range(1,N+1)]
            ,nb_inds=N)
        for sigma in itertools.permutations(range(1,N+1))
        for p in itertools.product(*[[False]+[True]*((l in Prime) and sigma[l-1]!=l) for l in range(1,N+1)]
        )
        ],nb_inds=N).sort()

def nbcycles(sigma):
    """
    Computes the number of disjoint cycles in a given permutation.

    Args:
        sigma (list of int): A list representing a permutation, where the
            i-th element is the position that the i-th index is mapped to.

    Returns:
        int: The number of disjoint cycles in the permutation.

    Explanation:
        - Starts with a list of all indices in `sigma`.
        - For each index, follows the cycle created by the permutation.
        - Removes visited indices from the list until all indices are processed.
    """
    ns=list(range(1,len(sigma)+1)) #list of all indices
    res=0
    while ns:
        res+=1
        i=ns[0]
        l=[sigma[i-1]]
        while (i not in l):
            l.append(sigma[l[-1]-1])
        [ ns.remove(k) for k in l ]
    return res

def dtype(A): return A.dtype if isinstance(A,sparse.COO) else "object"

def G(**args):
    args["method"]="J_matrix"
    return SymT(ReturnG=True,**args)

def chi(*largs,**args):
    """ 
    this one is explictely a chi of so
    synthax includes:
    chi(s,r) for a symmetric repr
    J_matrix(s,r,a) for a rectangular repr
    chi(Lambda,r) for a Young diag
    J_matrix(s,r,method=J_matrix), J_matrix(s,r,a,method=J_matrix), J_matrix(Lambda,r,method=J_matrix)
    """
    largs=list(largs)
    s,r,a,Lambda=(args.get(k,None) for k in ["s","r","a","Lambda"])
    na=len(largs)+sum(1 for k in (s,r,a,Lambda) if k!=None) # nb of arguments
    if na==2 and a==None:
        s,r=(largs.pop(0) if k==None else k for k in [(s if Lambda==None else Lambda),r])
        if hasattr(s, '__iter__'):s,Lambda=None,s
    elif na==3: s,r,a=(largs.pop(0) if k==None else k for k in [s,r,a])
    else: raise ValueError("unable to parse arguments of function chi")
    if s!=None and a==None: a=1
    repr={k:v for (k,v) in [("a",a),("s",s),("r",r),("Lambda",Lambda)] if v!=None}
    if args.get("method","")=="J_matrix":
        res=sparse.DOK(shape=sparse_J_matrix.J_matrix(1,1,**repr).shape,dtype="object")
        for k in range(res.shape[0]):
            res[k,k]=1
        for i in range(1,r+1):
            JM=sparse_J_matrix.J_matrix(i,i,**repr)
            assert is_diag(JM)
            for k in range(JM.shape[0]):
                res[k,k]*=x(i)**JM[k,k]
        res=tensor(res)
        return res.trace()
    elif a==1: return chis(s=s,r=r,grp="so")
    elif a==0: return 1
    elif s==1: return chis(s=s,r=r,grp="so",a=a)
    else: raise NotImplementedError

def SymT(N=3,s=None,**args):
    """ if s is specified, computes T_{1,s}, otherwise computes the generating series (up to a factor $w$)
    args (default) include:

    method ("generating series"): set to J_matrix in order to really take a trace
    Test(False -> 1/2/etc): temporary option to check simpler equalities
    gl (false): to compute a gl T-operator instead of so
    a (1): to compute T_{a,s} instead of T_{1,s}
 """
    if args.get("method",False)=="J_matrix": # See T_matrix in stuff.py
        assert sympy.sympify(s).is_integer and "r" in args
        assert not args.get("gl",False)
        assert not "gtens" in args
        r=args["r"]
        a=args.get("a",1)
        dt=dtype(sparse_J_matrix.J_matrix(1,1,s,r,a=args.get("a",1)))
        G=sum(tensor(sparse.DOK((2*r,2*r),{tuple(x+r-1 if x<=r else x-r-1 for x in (i,j)):1}).to_coo()).tensordot(tensor(sparse_J_matrix.J_matrix(j,i,s,r,a=args.get("a",1))),0) for (i,j) in itertools.product(*([range(1,2*r+1)]*2)))
        if args.get("ReturnG",False): return G
        if a==1:
            Lax=(lambda u:(u**2-((r-2)**2+2*(r-1)*s+s**2)/sympy.sympify(4))
              +u*G
              +(r-1)/sympy.sympify(2)*G
             +1/sympy.sympify(2)*(G**2)
             )
        else:
            sg=G.array.shape[-1]
            tg2=(G**2).trace((-2,-1)).tensordot(sparse.DOK((sg,sg),{(i,i):1 for i in range(sg)}),0)
            Lax=(lambda u:u**2-(r-2)**2/sympy.sympify(4)
              +u*G
              +(r-1)/sympy.sympify(2)*G
             +1/sympy.sympify(2)*(G**2)
             -1/sympy.sympify(8)*tg2
             )
        randargs=args.copy()
        rd=args.get("random",False)
        if rd==False:
            randargs["random"]=False
        else:
            randargs.pop("random")
            if rd!=True: randargs["dtype"]=rd
        ## produce the initial twist prod(x_i^{F_ii})
        res=sparse.DOK(shape=sparse_J_matrix.J_matrix(1,1,s,r,a=args.get("a",1)).shape,dtype="object")
        for k in range(res.shape[0]):
            res[k,k]=1
        for i in range(1,r+1):
            JM=sparse_J_matrix.J_matrix(i,i,s,r,a=args.get("a",1))
            assert is_diag(JM)
            for k in range(JM.shape[0]):
                res[k,k]*=formal.randsubs(x(i),**randargs)**JM[k,k]
        res=tensor(res)
        # res.print_nice()
        for k in range (1,N+1):
            res=tensor.tensordot(formal.randsubs(
                Lax(sympy.Function("u")(N+1-k)),**randargs),
                                 res,([-2],[-1])).transpose([0,1]+list(range(3,2*k+2))+[2])
            # the way indices are contracted is non-intuitive. It corresponds to writing
            # g⋅Lₙ⋅…⋅L₂⋅L₁ as a L₁×L₂×…×Lₙ×g where A×B=B⋅A (ie fancy contraction of indices)
            # Rq:  (trace is cyclic no matter whether g is rigt or left)
            # Rq: A×B=B⋅A tells how indices are contracted, but if matrix elements do not commute, it is rather A×B=AᵏBₖ (as opposed to A⋅B which is AₖBᵏ)
        # It requires also a trace at the end of the day
        res=res.trace() if N==0 else res.trace((-2,-1)) #take a trace
        return res
    r=args.get("r",sympy.Symbol("r"))
    dct={"1":(lambda i:u(i)**2-(r-2)**2/4),"D":(lambda i:u(i)+(r-1)/2),"D^2'":(lambda i:1/sympy.sympify(2)),"tr(D^2)":(lambda i:-1/sympy.sympify(8))}
    if args.get("notr",False):
        dct.pop("tr(D^2)")
    if args.get("free_coefs",False):
        c0,c1,c2,c3,c4=(sympy.Symbol("c_"+str(n)) for n in range(5))
        dct={"1":(lambda i:u(i)**2+c0),"D":(lambda i:c1*u(i)+c2),"D^2'":(lambda i:c3),"tr(D^2)":(lambda i:c4)}
    if args.get("gl",False):
        dct={"1":(lambda i:u(i)),"Dgl":(lambda i:1)} # convention of for GL operators arXiv:1206.4061
    gen=formal(terms=[(1 if args.get("gl",False) or "gtens" in args else (1-z**2))*pref*DW(term=trm,**{k:args[k] for k in ["gtens"] if k in args},**{"pow":-1 for i in ([1] if s==1 and args.get("a",1)>1 else [])})
        for trm in itertools.product(*([list(dct)]*N))
        for pref in [sympy.Mul(*[dct[t](n+1) for n,t in enumerate(trm)])]
    ]).sort()
    if type(s)==int and "gtens" not in args:
        tmp=gen.coefs(z,max(s,args.get("a",1))) #returns all terms z^.. up to the power z^s
        if args.get("a",1)==1: return sum(e*chis(s-p,grp="gl")
                   for (p,e) in enumerate(tmp))
        else: return sum(e*chis(s=1,a=args.get("a",1)-p,grp="so")*(-1)**p
                 for (p,e) in enumerate(tmp))
    else: return gen


def chis(s,r="symb",grp="so",a=1):
    """
    Computes a characteristic function or related expression based on the parameters.

    Args:
        s (int): A parameter related to the size or dimension of the function.
        r (str or int, optional): The rank or mode of operation. Default is "symb".
            - If "symb", the function returns a symbolic expression.
        grp (str, optional): The group type, either "so" (special orthogonal) or "gl" (general linear).
            Default is "so".
        a (int, optional): An additional parameter affecting the calculation. Default is 1.

    Returns:
        int or sympy.Expr: The computed value or symbolic expression based on the inputs.

    Details:
        1. If `a * s == 0` and `max(a, s) >= 0`, it returns 1 as a trivial case.
        2. For `r == "symb"`:
            - Returns either `chiso` or `chigl` (symbolic functions) based on the group type (`grp`).
        3. For `grp == "so"` and `a == 1`:
            - Computes the difference between the current and a reduced characteristic function.
        4. For `grp == "gl"` and `a == 1`:
            - Uses combinations to compute a sum over tensor products.
        5. For `a != 1`:
            - Computes a series expansion and extracts the coefficient for a specific power of `z`.

    Examples:
        - `chis(2, r=3, grp="gl", a=1)` computes a general linear group characteristic.
        - `chis(3, grp="so")` computes for a special orthogonal group.
    """
    if a*s==0 and max(a,s)>=0: return 1
    if r=="symb":
        return (chiso if grp.lower().startswith("so") else chigl)(*(([a] if a!=1 else [])+[s]))
    if grp.lower().startswith("so") and a==1 :return chis(s=s,r=r,grp="gl")-(chis(s=s-2,r=r,grp="gl") if s>=2 else 0)
    elif a==1:
        xx= lambda i:x(i) if i>0 else 1/x(-i)
        xr= lambda i:xx(-r+i+bool(i>=r))
        return sum(sympy.Mul(*[xr(b-a) for (a,b) in enumerate(p)])
            for p in itertools.combinations(range(2*r+s-1),s))
    else: return ((-1)**a*sympy.Mul(*[(1-z*x(k))*(1-z/x(k)) for k in range(1,1+r)]
                    )).series(z,0,a+1).coeff(z**a) # assumed to be the same for so and gl

class tensor:
    """
    A class to represent and manipulate tensors generated from formal expressions.

    The `tensor` class integrates symbolic and numerical computations using libraries
    like NumPy, SymPy, and Sparse, supporting operations on both dense and sparse tensors.

    Attributes:
        array: The internal representation of the tensor, which can be a dense NumPy array,
               a SymPy ImmutableSparseNDimArray, or a Sparse COO/DOK array.

    Methods:
        __init__: Initializes the tensor based on the input expression and arguments.
    """
    def __init__(self,expr,**args):
        """
        Initializes a `tensor` object based on the provided formal expression and additional parameters.

        Args:
            expr: The input expression. Can be of the following types:
                - `formal`: A symbolic representation of the tensor.
                - `numpy.ndarray`, `sparse.COO`, or `sympy.ImmutableSparseNDimArray`: A tensor-like structure.
                - `sparse.DOK`: A dictionary of keys representation for sparse tensors.
                - `tensor`: Another `tensor` instance.
            **args: Additional keyword arguments controlling tensor initialization:
                - `r` (int): The rank of the tensor (required if `expr` is `formal`).
                - `random` (bool or dict): Enables random substitutions. Can specify data types.
                - `randinit` (bool): If True, resets the random substitution cache.
                - `dtype` (str): Specifies the data type for random initialization.
                - `SympyDefault` (bool): Uses SymPy-based sparse arrays for symbolic tensors.

        Raises:
            ValueError: If the input expression is not valid or required arguments are missing.

        Workflow:
            1. Handles initialization of `formal` expressions using symbolic or random substitutions.
            2. Converts sparse data structures (`DOK`, `COO`) to appropriate internal formats.
            3. Creates dense or sparse representations based on the input and arguments.

        Example:
            # Initialize a tensor from a formal expression
            t = tensor(formal_expr, r=2, random=True)
        """
        # Initialize random substitution if requested
        if args.get("randinit",False):
            formal.randinit()
            args.pop("randinit")
        
        # Prepare arguments for random substitutions
        randargs=args.copy()
        rd=args.get("random",False)
        if rd==False:
            randargs["random"]=False
        else:
            randargs.pop("random")
            if rd!=True: randargs["dtype"]=rd
        if "r" in args: randargs["r"]=args["r"]

        # Handle SymPy as the default backend for sparse arrays
        if tensor.SympyDefault and isinstance(expr,sparse.COO):
            expr=sympy.ImmutableSparseNDimArray(expr.todense())

        # Case 1: Input is a supported array type
        if isinstance(expr,(numpy.ndarray,sparse.COO,sympy.ImmutableSparseNDimArray)):
            self.array=formal.randsubs(expr,**randargs) if args.get("random",False) else expr
            return None
        
        # Case 2: Input is another tensor
        if isinstance(expr,tensor):
            self.array=expr.array
            return None
        
        # Case 3: Input is a sparse DOK array
        if isinstance(expr,sparse.DOK):
            self.array=(expr.to_coo() if expr.dtype!=numpy.dtype("object") and not tensor.SympyDefault else SparsAr(expr.data,expr.shape))
            return None
        
        # Case 4: Input must be a `formal` expression with a valid rank `r`
        if not (type(args.get("r",False))==int and isinstance(expr,formal)):
            tensor.tmp=expr
            raise ValueError("Needs a 'formal' expr and a value of 'r'")
        
        # Initialize tensor from a formal expression
        r=args["r"]
        tmptr={}
        randargs["tmptr"]=tmptr
        nbinds=expr._terms_[0].nb_inds if expr.issum else expr.nb_inds

        # Determine data type for the tensor
        random=args.get("random",False)
        if not isinstance(random,(dict,bool)): random={"All":random}
        dt=("object" if random==False or (type(random)==dict and any(random[k]=="Q" for k in random)) else ("int" if (type(random)==dict and all(random[k].startswith("int") for k in random)) else "float128"))
        
        # Create a sparse dictionary of keys (DOK) for the tensor data
        res=sparse.DOK((2*r,)*(2*nbinds),dtype=dt,fill_value=0)

        # Process each term in the formal expression
        for t in expr.terms():
            pref=t.pref()
            pref=formal.randsubs(pref,**randargs) # Apply random substitutions if required

            # Compute diagonals for factors
            diags=[
                numpy.array([formal.randsubs(f,g=ev,**randargs) for ev in [1/x(-k) for k in range(-r,0)]+[x(k) for k in range(1,r+1)]],dtype="object")
                for (i,j,f) in t.facs()]
            
            # Compute tensor entries by iterating over combinations
            for II in itertools.product(*([range(2*r)]*t.nb_inds)):
                JJ=[2*r]*(2*nbinds)
                for n,(i,j,f) in enumerate(t.facs()):
                    JJ[2*(i-1) if i>0 else 1+2*(abs(i)-1)]=II[n] if i>0 else 2*r-1-II[n]
                    JJ[2*(abs(j)-1) if j<0 else 1+2*(j-1)]=II[n] if j>0 else 2*r-1-II[n]
                JJ=tuple(JJ)
                
                res[JJ]+=sympy.prod((diags[n][i] for n,i in enumerate(II)),pref)  #the product of these diagonal elements

        # Convert the result to the appropriate format
        if dt=="object":
            self.array=SparsAr(res.data,res.shape)
        else:
            self.array=res.to_coo()
        return None

    def _latex_(self,brackets=True):
        """
        Generates the LaTeX representation of the tensor for rendering in Jupyter Notebooks.

        Args:
            brackets (bool, optional): Determines if parentheses should surround the LaTeX output.
                - True: Adds parentheses.
                - False or None: No parentheses.

        Returns:
            str: The LaTeX representation of the tensor as a string.

        Workflow:
            - If the tensor is 2D, it uses standard LaTeX matrix formatting.
            - For higher dimensions, it includes separators for rows (`\\ \hline`) and uses recursive LaTeX formatting.

        Example:
            - Display in a Jupyter Notebook using:
            ```python
            display(Markdown("$" + tensor_instance._latex_() + "$"))
            ```
        """
        if brackets: return r"\left("+self._latex_(brackets=None)+r"\right)"
        if isinstance(self,(numpy.ndarray,sparse.DOK,sparse.COO,sympy.ImmutableSparseNDimArray)): a=self
        else: a=self.array
        shp=a.shape
        assert len(shp)>=2
        v,h,f=("",r"\\",sympy.latex) if len(shp)==2 else ("|",r"\\ \hline ",lambda i:tensor._latex_(i,brackets=False))
        return r"\begin{array}{"+v.join(["c"]*shp[1])+"}"+" "+h.join(" & ".join(
            f(c) for c in l) for l in a)+r"\end{array}"

    def print_nice(self):
        """
        Prints the tensor in a visually appealing format using LaTeX.

        Workflow:
            - If the LaTeX string is short (<4000 characters), it is directly displayed in the notebook.
            - For longer outputs, it falls back to a generic method from the `formal` class.
        """
        X=self._latex_()
        if len(X)<=4000:
            display(Markdown("$"+X+"$"))
        else:
            formal.print_nice(X)
    def __getitem__(self,k): return self.array.__getitem__(k)
    def __tosametypeas__(self,B):
        """
        Converts the tensor and another tensor `B` to the same type for operations.

        Args:
            B: Another tensor to align types with.

        Returns:
            tuple: A pair of tensors `(self, B)` converted to the same underlying type.
        """
        if any(all(isinstance(o.array,t) for o in (self,B)) for t in (numpy.ndarray,sparse.COO,sympy.ImmutableSparseNDimArray)):
            return (self,B)
        elif any(isinstance(o.array,t) for o in (self,B) for t in [numpy.ndarray]):
            return tuple([tensor(numpy.array(o.array)) for o in (self,B)])
        elif any(isinstance(o.array,t) for o in (self,B) for t in [sympy.ImmutableSparseNDimArray]):
            return tuple([o if isinstance(o.array,sympy.ImmutableSparseNDimArray) else #"else" means sparse.COO
                          tensor(SparsAr(sparse.DOK(o.array).data,o.array.shape))
                          for o in (self,B)])
        else: 1/0
    def __mul__(self,B): # multiplication of tensors that both have the ijij indexing
        if type(B) in [int]:
            return tensor(B*self.array)
        if not isinstance(B,tensor):
            if isinstance(B,(numpy.ndarray,sparse.COO,sympy.ImmutableSparseNDimArray)):
                return self*tensor(B)
            if isinstance(self.array,sparse.COO):
                X=sparse.COO(self.array.coords,self.array.data*B,self.array.shape)
                if not str(X.dtype).startswith("int"): X=SparsAr(sparse.DOK(X).data,X.shape)
                return tensor(X)
            elif isinstance(self.array,sympy.ImmutableSparseNDimArray):
                return tensor(B*self.array)
            elif hasattr(B,"subs"): #most probably a sympy expression
                if isinstance(self.array,(numpy.ndarray,sympy.ImmutableSparseNDimArray)):
                    return tensor(B*self.array)
            raise NotImplementedError
        (C,D)=self.__tosametypeas__(B)
        nbi=len(C.array.shape)//2
        if isinstance(C.array,(numpy.ndarray,sparse.COO)): #sparse or numpy functions
            if isinstance(C.array,sparse.COO):
                tensordot,transpose=sp_tensordot,sp_transpose
            else: tensordot,transpose=np_tensordot,np_transpose
            return tensor(transpose(
            tensordot(transpose(C.array,[2*k+i for i in range(2) for k in range(nbi)]),
                      transpose(D.array,[2*k+i for i in range(2) for k in range(nbi)])
                ,nbi)
                ,[k+nbi*i for k in range(nbi) for i in range(2)]))
        else: # sympy functions
            A=sympy.tensorproduct(C.array,D.array)
            for i in range(nbi):
                A=sympy.tensorcontraction(A,(i+1,2*nbi))
            return tensor(sympy.permutedims(A,[k+nbi*i for k in range(nbi) for i in range(2)]))
    def trace(self,AX=None):
        """
        Computes the trace of the tensor by contracting specified pairs of axes.

        Args:
            AX (tuple, optional): A pair of axes to contract. If not provided, contracts the first two axes
                                (assuming the tensor is 2D).

        Returns:
            tensor: A new tensor resulting from the trace operation.

        Workflow:
            - If `AX` is specified, it determines the axes for contraction.
            - For NumPy arrays, uses `trace` for efficient contraction.
            - For Sparse COO arrays, computes the trace using slice and summation.
            - For SymPy arrays, uses `tensorcontraction`.

        Example:
            ```python
            result = tensor_instance.trace(AX=(-2, -1))  # Contract the last two axes
            ```
        """
        A=self.array
        if not AX: assert len(A.shape)==2
        if AX:
            n=len(A.shape)
            AX=[n+i if i<0 else i for i in AX]
        if isinstance(A,numpy.ndarray):
            if AX: tensor(A.trace(axis1=AX[0],axis2=AX[1]))
            else: return A.trace()
        elif isinstance(A,sparse.COO):
            if AX: return tensor(sum(A.__getitem__(tuple([(slice(None) if k not in AX else i) for k in range(len(A.shape))])) for i in range(A.shape[0])))
            else: return sum(A[i,i] for i in range(A.shape[0]))
        else:
            if AX: return tensor(sympy.tensorcontraction(A,AX))
            else: return sympy.tensorcontraction(A,(0,1))
    def __add__(self,B): 
        """
        Adds two tensors element-wise.

        Args:
            B (tensor or scalar): The second tensor to add. Scalars are broadcast to tensors of the same shape.

        Returns:
            tensor: A new tensor representing the sum.

        Workflow:
            - Converts scalars to tensors if necessary.
            - Ensures both tensors have the same type using `__tosametypeas__`.
            - Performs addition based on the underlying type (NumPy, Sparse, or SymPy).

        Example:
            ```python
            result = tensor1 + tensor2
            ```
        """
        if not isinstance(B,tensor):
            TmpT=sparse.DOK(self.array.shape,{tuple(i for i in II for k in range(2)):B
                                          for II in itertools.product(*[range(k) for k in self.array.shape[::2]])})
            return self+tensor(TmpT)
        (C,D)=self.__tosametypeas__(B)
        return tensor(C.array+D.array)
    def __sub__(self,B):
        """
        Subtracts two tensors element-wise.

        Example:
            ```python
            result = tensor1 - tensor2
            ```
        """
        return self+(-1*B)
    def __neg__(self): return -1*self
    def __pow__(self,po):
        """
        Raises the tensor to the power `po` using repeated multiplication.

        Args:
            po (int): The power to raise the tensor to. Must be non-negative.

        Returns:
            tensor: A new tensor representing the power.

        Workflow:
            - Handles `po == 0` by returning an identity tensor.
            - Splits the computation into `x**(2*a+b) = (x**a)**2 * x**b` for efficiency.

        Example:
            ```python
            result = tensor_instance ** 3
            ```
        """
        assert po>=0
        if po==0: return tensor(sparse.DOK(self.array.shape,{tuple(i for i in II for k in range(2)):1
                                          for II in itertools.product(*[range(k) for k in self.array.shape[::2]])}).to_coo())
        a,b=(po//2,po%2) 
        sq=next(s*s for s in [self**a])
        if b: return self*sq
        else: return sq
    def tensordot(self,B,n):
        """
        Computes the tensor dot product along specified axes.

        Args:
            B (tensor): The second tensor.
            n (int or list of tuples): Number of axes to contract or explicit axes for contraction.

        Returns:
            tensor: A new tensor resulting from the dot product.

        Workflow:
            - Aligns types of tensors using `__tosametypeas__`.
            - Handles contraction for SymPy arrays using `tensorcontraction`.
            - Uses NumPy or Sparse methods for efficient computation when appropriate.

        Example:
            ```python
            result = tensor1.tensordot(tensor2, n=2)  # Contract along the last 2 axes of tensor1 and first 2 of tensor2
            ```
        """
        if not isinstance(B,tensor): B=tensor(B)
        (C,D)=self.__tosametypeas__(B)
        if isinstance(C.array,sympy.ImmutableSparseNDimArray):
            X=sympy.tensorproduct(C.array,D.array)
            Nc=len(C.array.shape)
            if not hasattr(n,"__iter__"):
                n=(list(range(-n,0)),list(range(n)))
            n=[[k if k>=0 else k+len((C,D)[x].array.shape) for k in k] for x,k in enumerate(n)]
            while len(n[0]):
                X=sympy.tensorcontraction(X,(n[0][0],Nc+n[1][0]))
                n=[[k-(1 if k>=y[0] else 0) for k in y[1:]] for y in n]
            return tensor(X)
        if isinstance(C.array,sparse.COO):
            tensordot,transpose=sp_tensordot,sp_transpose
        if isinstance(C.array,numpy.ndarray):
            tensordot,transpose=np_tensordot,np_transpose
        if not hasattr(n,"__iter__"):
            return tensor(tensordot(C.array,D.array,n))
        else:
            return tensor(tensordot(
                C.array.transpose([k for k in range(len(C.array.shape)) if k not in n[0]]+n[0]),
                D.array.transpose(n[1]+[k for k in range(len(D.array.shape)) if k not in n[1]]),
                len(n[0])))
    def g(**args):return tensor(formal(g),**args)
    def replace(self,*k,**l):
        """
        Replaces elements in the tensor using specified substitutions.

        Args:
            *k: Positional arguments passed to the `replace` method of individual elements.
            **l: Keyword arguments passed to the `replace` method of individual elements.

        Returns:
            tensor: A new tensor with substitutions applied.

        Workflow:
            - For Sparse COO tensors, no replacement is needed since they are integer-based.
            - For SymPy-based tensors, replacements are applied to individual sparse elements.
            - For NumPy arrays:
                - If the array has object (`O`) dtype, replacements are applied to each element.
                - Otherwise, the tensor is returned unchanged.

        Raises:
            NotImplementedError: If the tensor format is unsupported.

        Example:
            ```python
            substituted_tensor = tensor_instance.replace(sympy.Symbol('x'), sympy.Symbol('y'))
            ```
        """
        if isinstance(self.array,sparse.COO): return self #just integers, nor replacement to be done
        elif isinstance(self.array,sympy.ImmutableSparseNDimArray):
            return tensor(SparsAr({i:d[i].replace(*k,**l) for d in [self.array._sparse_array] for i in d},self.array.shape))
        elif isinstance(self.array,numpy.ndarray):
            if self.array.dtype==numpy.dtype('O'):
                t=self.array.copy()
                for II in itertools.product(*([range(k) for k in self.array.shape])):
                    t.__setitem__(II,t[II].replace(*k,**l))
                return tensor(t)
            else: return self
        else: raise NotImplementedError
    def transpose(self,*I):
        """
        Permutes the axes of the tensor.

        Args:
            *I:
                - If a single list is provided (e.g., `[2, 3, 1, 0]`), it performs a straightforward axis permutation.
                - If two lists are provided (e.g., `[2, -1], [2, -1]`), it interprets them as distinct "I" and "J" axes
                for permuting rows and columns.

        Returns:
            tensor: A new tensor with axes permuted.

        Workflow:
            - Converts the two-list format to a single list for permutation.
            - Applies the appropriate transpose function based on the tensor's type:
                - `sp_transpose` for Sparse COO.
                - `np_transpose` for NumPy arrays.
                - `sympy.permutedims` for SymPy-based tensors.
            - If two lists are provided, the lines/columns are permuted to handle sign changes due to negative indices.

        Raises:
            NotImplementedError: If the tensor format is unsupported.

        Example:
            ```python
            # Standard axis permutation
            transposed_tensor = tensor_instance.transpose([2, 3, 1, 0])

            # Row-column interpretation
            transposed_tensor = tensor_instance.transpose([2, -1], [2, -1])
            ```
        """
        if len(I)==2: # Convert row-column format to a single list
            nI=[2*nb-2+k if nb>0 else 2*abs(nb)-2+(1-k)
                for i in range(len(I[0]))
                for k in [0,1] # k differentiates between i and j
                for nb in [I[k][i]]]
        else: nI=I[0] # Single-list format

        if isinstance(self,tensor):
            self=self.array

        # Select the appropriate transpose method
        if isinstance(self,sparse.COO): transpose=sp_transpose
        elif isinstance(self,numpy.ndarray): transpose=np_transpose
        elif isinstance(self,sympy.NDimArray): transpose=sympy.permutedims
        else: raise NotImplementedError
        self=transpose(self,nI)

        # Adjust for negative indices in the two-list format
        if len(I)==2:
            self=self.__getitem__(tuple([slice(None,None,1 if (I[k][n])>0 else -1)
                                         for n in range(len(I[0])) for k in range(2)]))
        return tensor(self)
        



        
    
tensor.__rmul__=tensor.__mul__ #if one of them is not a "tensor", then they commute
tensor.__radd__=tensor.__add__

tensor.SympyDefault=True

        
def coefs(e,v,p):
    """
    Computes the coefficients of a given variable `v` up to power `p` in the expression `e`.

    Args:
        e (formal, sympy.Expr, or other types): The input expression.
        v (sympy.Symbol): The variable for which coefficients are computed.
        p (int): The highest power of `v` for which coefficients are computed.

    Returns:
        list: A list containing coefficients for powers of `v` from 0 to `p`.

    Notes:
        - Handles various types of expressions including `formal`, `sympy.Add`, `sympy.Mul`, and custom types like `gzrat` and `tr`.
        - Includes special handling for cases like rational expressions or variables raised to negative powers.
    """
    # Handle `formal` expressions
    if isinstance(e,formal):return e.coefs(v,p)
    # Handle expressions without the variable `v`
    if isinstance(e,sympy.Expr) and not e.has(v):#sympy expression that does not contain v
        return [e]+[0]*p
    # Handle `gzrat` type
    if type(e)==gzrat:
        if v!=e.args[0]: return [0]*(p+1)
        return [0 if k==0 and all(a.name!="shift" for a in e.args[1:]) else G**k
                for G in [g if all(a.name!="prim" for a in e.args[1:]) else 1/g]
                for k in range(p+1)]
    # Handle `tr` type
    if type(e)==tr: return [
        tr(t) if (type(t)!=type(sympy.sympify(0)) if hasattr(t,"subs") else t!=0)
        else 0
        for t in coefs(e.args[0],v,p)]
    # Handle addition (`sympy.Add`)
    if type(e)==sympy.Add:
        return [ sum(l[k] for l in lc)
            for lc in [[coefs(t,v,p) for t in e.args]]
            for k in range(p+1)]
    # Handle multiplication and positive integer powers
    if type(e)==sympy.Mul or (type(e)==sympy.Pow and e.args[1].is_Integer and e.args[1]>=0):
        facts=e.args if type(e)==sympy.Mul else [e.args[0]]*e.args[1]
        cfact=[coefs(f,v,p) for f in facts]

        return [sum(sympy.Mul(*[f[ds.count(i)] for (i,f) in enumerate(cfact)])
            for dis in itertools.combinations(range(len(facts)+s-1),s)
                    for ds in [[b-a for (a,b) in enumerate(dis)]])
            for s in range(p+1)
        ]
    # Handle single variables
    if e==v: return [0] if p==0 else [0,1]+[0]*(p-1)
    # Handle numeric values
    if (hasattr(e,"is_number") and e.is_number) or hasattr(e,"real") or type(e) in [u] or e in [r]:
        return [e]+[0]*p
    # Handle rational expressions of the form 1 / (A + v * B)
    if type(e)==sympy.Pow and e.args[1].is_Integer and e.args[1]<0 and e.args[0]==(e.args[0].subs({v:0})+v*e.args[0].coeff(v)) and e.args[0].subs({v:0})!=0: # 1/(A+v*B)
        A,B=e.args[0].subs({v:0}),e.args[0].coeff(v)
        lp=[(-1)**k*B**k/A**(k+1) for k in range(p+1)]
        if e.args[1]==-1: return lp
        else: return coefs(sympy.Mul(*([sum(lp[k]*v**k for k in range(p+1))]*(-e.args[1]))),v,p)
    formal.tmp=e
    return 1/0

# Tensor-related methods
tensor.randsubs=formal.randsubs
formal.P=formal(factors=[(1,2,1),(2,1,1)],nb_inds=2)
formal.I=formal(factors=[(1,1,1)],nb_inds=1)
formal.Q=formal(factors=[(-1,2,1),(1,-2,1)],nb_inds=2)


def is_diag(M):
    """
    Checks if a matrix is diagonal.

    Args:
        M (tensor, sympy.ImmutableSparseNDimArray, or numpy.ndarray): The matrix to check.

    Returns:
        bool: True if the matrix is diagonal, False otherwise.
    """
    if isinstance(M,tensor):M=M.array
    if len(M.shape)!=2: raise NotImplementedError
    k=M.shape[0]
    if M.shape[1]!=k: return False
    if isinstance(M,sympy.ImmutableSparseNDimArray): return all((i%k)==(i//k) for i in M._sparse_array)
    if isinstance(M,sparse.DOK): return all(i[0]==i[1] for i in M.data)
    if isinstance(M,sparse.COO): return all(M.coords[0]==M.coords[1])
    if isinstance(M,numpy.ndarray): return numpy.array_equal(M, numpy.diag(numpy.diag(M)))
    return NotImplementedError



def LeadingT(a,s,r,**args):
    """
    Computes the coefficient of the highest-order term of T_{a,s}, as described in equation (8.29).

    Args:
        a (int): A parameter related to the rank of the determinant.
        s (int): A parameter related to symmetry or shifts in indices.
        r (int): The rank or size parameter for the calculation.
        **args: Additional optional arguments:
            - random (bool or dict): If `True`, performs random substitutions for symbolic variables.
                                     If `False`, does not perform substitutions.
                                     If a dictionary, specifies dtype or substitution rules.

    Returns:
        sympy.Expr: The computed coefficient of the highest-order term.

    Notes:
        - Utilizes the symmetry of indices and determinants of symbolic matrices.
        - Uses equation (8.15) to define the `h(i)` function for calculations.
        - Combines symbolic matrix determinants to compute the final result.

    Helper Functions:
        - `h(i)`: Computes the product of 1 / (x(i) + 1/x(i) - (x(j) + 1/x(j))) for all `j != i`.
        - `X(i)`: Performs substitution of `x(i)` or `1/x(-i)` depending on the sign of `i`.
    """
    # Check for random substitution arguments
    random=args.get("random",False)
    rdd=(({} if random else {"random":False}) if type(random)==bool else {"dtype":random})

    def h(i): #defined right after (8.15)
        i=abs(i) #i' and i have the same h
        return sympy.Mul(*[formal.randsubs(1/((x(i)+1/x(i))-(x(j)+1/x(j))),**rdd) for j in range(1,r+1) if j!=i])
    def X(i): return formal.randsubs(x(i) if i>0 else 1/x(-i),**rdd)
    return sum(sympy.Mul(*[h(i) for i in i_k ]
                         )*sympy.Matrix([[X(i_a)**((a+1-2*b+s+r-1)/sympy.sympify(2)) for i_a in i_k ] for b in range(1,a+1)]).det(
                         )*sympy.Matrix([[X(-i_a)**((a+1-2*b-s-r+1)/sympy.sympify(2)) for i_a in i_k ] for b in range(1,a+1)]).det()
        for i_k in itertools.combinations(list(range(-r,0))+list(range(1,r+1)),a))



def su_str(e,m="₋",z="₀"):
    """
    Converts an integer to a subscript or superscript string.
    """
    dct={c:bytes(k+(ord(c)-ord("0") if n==2 else 0) for n,k in enumerate(bytes(z,"utf-8"))).decode("utf-8") for c in "0123456789"}
    if z=="⁰":dct["1"],dct["2"],dct["3"]="¹²³"
    dct["-"]=m
    return "".join(dct[p] for p in str(e))
def substr(e):
    """Shortcut for subscript formatting."""
    return su_str(e,"₋","₀")
def supstr(e): 
    """Shortcut for superscript formatting."""
    return su_str(e,"⁻","⁰")

def reformat(s):
    r""" converts typically \lambda_{-1}^{[1]} into λ₋₁⁽¹⁾ """
    while r"\lambda_" in s:s=s.replace(r"\lambda_","λ_")
    while "_{" in s:
        k=s.index("_{")
        l=s[k:].index("}")
        s=s[:k]+substr(s[k+2:k+l])+s[k+l+1:]
    while "^{[" in s:        
        k=s.index("^{[")
        l=s[k:].index("]}")
        s=s[:k]+"⁽"+supstr(s[k+3:k+l])+"⁾"+s[k+l+2:]
    return s


class Lprod:
    def __init__(self,*m,**args):
        """
        Initializes an Lprod object, which represents a formal product of terms.

        Examples:
            - `Lprod(listops=[["L", 1, 2, u], ["L", 2, 3, u+1]])`
            - `Lprod(terms={(("L", 1, 2, u), ("L", 2, 3, u+1)): 1, (): 2})`
            - `Lprod("Lk", 1, 2, u, r=3)`

        Arguments:
            *m: Input terms or initialization values.
            **args:
                - listops: A list of operations defining the terms.
                - terms: A dictionary mapping tuples of symbolic terms to coefficients.
                - r: A parameter that defines additional properties of the terms.

        Notes:
            - "Lc" specifies \( L^i_j \), while "Lk" specifies \( L_i^j \) (Karakhanyan-Kirschner notation).
            - Expansion in powers of \( u \) is also supported via "Lo" or "Lluo".
            - `expand()` transforms \( L(u) \) into a series form like \( u^2 + uL^1(u) + L^2(u) \).
        """
        listops,terms,r=(args.get(k,None) for k in ["listops","terms","r"])
        self.r=r

        if terms:
            assert not(listops or m)  # Ensure terms are the only input if provided.
            self._terms=terms
            return None
        
        if not listops:
            if type(m[0])==dict:
                self._terms,=m
                return None
            elif hasattr(m[0],"__iter__"):
                if m[0]==tuple([]):
                    listops=[]
                else: 
                    lvl= next(k for k in range(1,5) if type(functools.reduce(lambda x,y:x[y],[m]+[0]*k))==str)
                    if   lvl==1: listops=[m]
                    elif lvl==2: listops=list(m)
                    elif lvl==3: listops=m[0]
                    else: raise ValueError
            else:
                v,=m
                self._terms={} if v==0 else {tuple():v}
                return None
            
        # Populate _terms dictionary based on listops
        self._terms={tuple([(n[0]+({"":"ul","k":"ll"}.get(n[1:],n[1:]))+("o" if k[0].endswith("o") else ""),)+tuple(k)[1:] for k in listops for n in [k[0].rstrip("o")]]):1}

    def __eq__(self,fles):
        """
        Equality comparison for Lprod objects.
        """
        if not isinstance(fles, Lprod):
            if fles==0: return all(self._terms[x]==0 for x in self._terms)
            return False
        
        if self.r!=fles.r: return False
        terms=[{x for x in O._terms if O._terms!=0} for O in [self,fles]]

        if terms[0]!=terms[1]: return False
        return all(self._terms[x]==fles._terms[x] for x in terms[0])
    
    def __hash__(self):
        """
        Hash function for Lprod objects.
        """
        return (self.r,tuple([(x,self._terms[x]) for x in self._terms if self._terms[x]!=0])).__hash__()
    
    def __repr__(self,extra=""):
        """
        String representation of Lprod objects.
        """
        if all (self._terms[k]==0 for k in self._terms): return "0"
        if extra and self._terms=={tuple():1}: return extra
        res= ("".join(("+" if s[0] !="-" else "")+(("-" if self._terms[k]<0 else "") if abs(self._terms[k])==1 and k!=tuple() else s)+
                       "".join(L+S(i)+C+T(j)+U for (l,i,j,*u) in k
                               for L in [l[0]] for S,T in [[{"u":supstr,"l":substr}[t] for t in l[1:3]]]
                               for C in ["," if l[1:3]=="ll" else ("⸴" if l[1:3]=="uu" else "")]
                               for U in ["" if l[0]in"GH" else ("("+str(u[0])+")" if l[-1]!="o" else "⁽"+supstr(u[0])+"⁾")]
                               )+extra
                           for k in self._terms if not self._terms[k]==0
                           for s in [str(self._terms[k])])).lstrip("+")
        return reformat(res)
    
    def __add__(self,B):
        """
        Overloads the addition operator for Lprod objects.
        """
        if not isinstance(B,Lprod): B=Lprod(B)
        rs={X.r for X in [self,B] if X.r!=None}
        if len(rs)==2: raise ValluerError("cannot add L with distinct r")
        if len(rs)==1: r,=rs
        if len(rs)==0: r=None
        A=set(self._terms).union(set(B._terms))
        D={k:sum(t._terms.get(k,0) for t in [self,B]) for k in A}
        return Lprod({k:D[k] for k in D if D[k]!=0},r=r)
    
    def __mul__(self,B):
        """
        Overloads the multiplication operator for Lprod objects.
        """
        if isinstance(B,Lvect):return B.__rmul__(self)
        if not isinstance(B,Lprod): B=Lprod(B)
        rs={X.r for X in [self,B] if X.r!=None}
        if len(rs)==2:
            Lprod.tmp=(self,B)
            raise ValueError("cannot multiply L with distinct r")
        if len(rs)==1: r,=rs
        if len(rs)==0: r=None
        R={}
        for x in self._terms:
            for y in B._terms:
                R[x+y]=R.get(x+y,0)+self._terms[x]*B._terms[y]
        return Lprod(R,r=r)
    
    def __sub__(self,B):
        """
        Overloads the subtraction operator for Lprod objects.
        """
        return self+(-1*B)
    
    def __neg__(self): 
        """
        Negation of Lprod objects.
        """
        return -1*self
    
    def copy(self):
        """
        Creates a copy of the Lprod object.
        """
        return Lprod(self._terms.copy(),r=self.r)
    
    def into(self,ind,inplace=True):
        """
        Adjusts index structure.

        Args:
            ind (str): Target index structure. Examples: "Lc", "Lk", "ul".
            inplace (bool): Whether to modify the instance in place.

        Returns:
            Lprod: Modified instance with transformed indices.
        """
        ind=ind.lstrip("L") # Remove the "L" prefix, if any.
        ind={"":"ul","c":"ul","k":"ll"}.get(ind,ind)  # Map shorthand notations.
        dct={}
        for k in self._terms:
            nk=tuple((l[0]+ind+l[3:],(i if l[1]==ind[0] else -i),(j if l[2]==ind[1] else -j),*u)
                       for (l,i,j,*u) in k)
            dct[nk]=dct.get(nk,0)+self._terms[k]
        if inplace: self._terms=dct
        else:  self=Lprod(dct,r=self.r)
        return self
    
    def Lcom(a,b,**args):
        """
        Computes the commutator [A, B].

        Args:
            a, b: `Lprod` instances or tuples representing terms.
            **args: Additional options:
                - r: Integer, parameter related to the commutator.
                - diff: Bool, if True, computes LHS - RHS of the commutator.
                - rev: Bool, if True, reverses the commutator [B, A].
                - sort: Bool, if True, sorts terms for easier interpretation.

        Returns:
            Lprod: The result of the commutator operation.
        """
        r,diff,rev,sort=(args.get(k,v) for (k,v) in [("r",None),("diff",False),("rev",False),("sort",False)])
        
        # Validate or deduce parameter `r`.
        r,=set(([] if r==None else [r])+[x.r for x in [a,b] if hasattr(x,"r") and x.r!=None]) #will produce an error message if r is not specified, because r is required for commutator

        # Convert inputs to tuples if necessary.
        a,b=[tuple(k) if hasattr(k,"__iter__") else
             (next(p[0] for p in k._terms if len(k._terms)==1 and k._terms[p]==1 and len(p)==1) if isinstance(k,Lprod) else ValueError("wrong format")) for k in [a,b]]
        args["r"]=r

        # Handle differentiation option.
        if diff:
            c=args.copy()
            c["diff"]=False
            R=Lprod.Lcom(a,b,**c)
            return Lprod(a,r=r)*Lprod(b,r=r)-Lprod(b,r=r)*Lprod(a,r=r)-R
        
        # Handle sorting option.
        if sort:
            # if any(
            c=args.copy()
            c["sort"]=False
            R=Lprod.Lcom(a,b,**c)
            tt=R._terms
            easy,HH={},{}
            for t in tt:
                if len(t)==1 or (len(t)==2 and ({t[0][0][0],t[1][0][0]}=={"G","H"} or (t[0][0]==t[1][0]=="Lllo" and {t[0][3],t[1][3]}=={1,2}))):
                    easy[t]=tt[t]
                elif len(t)==2 and ({t[0][0][0],t[1][0][0]}=={"H"} or (t[0][0]==t[1][0]=="Lllo" and {t[0][3],t[1][3]}=={2})):
                    HH[t]=tt[t]
                else:
                    Lprod.tmp=(tt,t)
                    raise ValueError
            R=Lprod(easy,r=r).sort()+Lprod(HH,r=r)
            OK,toswap={},{}
            for t in R._terms:
                if Lprod.toswap([t]): toswap[t]=R._terms[t]
                else:                 OK[t]  =R._terms[t]
            R.OK=Lprod(OK,r=r)
            R.toswap=Lprod(toswap,r=r)
            return R
        
        # Handle reverse commutator option.
        if rev:
            c=args.copy()
            c["rev"]=False
            return -Lprod.Lcom(b,a,**c)
        
        # Adjust indices if necessary.
        if any(k[0][1:3]!="ll" for k in [a,b]):
            return Lprod.Lcom(*([Lprod(k,r=r).into("ll") for k in [a,b]]),**args)
        
        if next(k[0][0]=="G" or (k[0][-1]=="o" and k[-1]==1) for k in [a]): #[G,H] or [G,G] or [G,L]
                # use (2.8) which contains (2.7) etc
                return (-Lprod(b[:1]+(a[1],)+b[2:],r=r) if b[1]+a[2]==0 else 0)+(Lprod(b[:2]+(a[2],)+b[3:],r=r) if a[1]+b[2]==0 else 0)+(Lprod(b[:1]+(a[2],)+b[2:],r=r) if a[1]+b[1]==0 else 0)+(-Lprod(b[:2]+(a[1],)+b[3:],r=r) if a[2]+b[2]==0 else 0)
        
        if next(k[0][0]=="G" or (k[0][-1]=="o" and k[-1]==1) for k in [b]): #[H,G] or [L,G]
                return -Lprod.Lcom(b,a,r=r)
        
        if any(len(k)<4 or k[0][0]!="L" or k[0][-1]=="o" for k in [a,b]):
            if a[0][0]=="L" and a[0][-1]!="o":
                return sum(a[-1]**p*Lcom((a[0][0]+"o",a[1],a[2],2-p) if b[0][-1]=="o" else
                                         (("G" if p==1 else "H"),a[1],a[2])
                                             ,b,r=r) for p in range(2) # used to be 3 but L⁽⁰⁾ doesn't contribute
                    )
            
            elif b[0][0]=="L" and b[0][-1]!="o":
                return sum(b[-1]**p*Lcom(a,(b[0][0]+"o",b[1],b[2],2-p) if a[0][-1]=="o" else
                                         (("G" if p==1 else "H"),b[1],b[2])
                                         ,r=r) for p in range(2)# used to be 3 but L⁽⁰⁾ doesn't contribute
                    )
            
            else :# both are H
                assert all(k[0][0]=="H" or (k[0][0]+k[0][-1]=="Lo" and k[-1]==2) for k in [a,b]) #both are H

                # use (5.1)
                if len(a)==3: G,H=(lambda i,j: Lprod("Gk",i,j,r=r)),(lambda i,j: Lprod("Hk",i,j,r=r))

                else: G,H=(lambda i,j: Lprod("Lko",i,j,1,r=r)),(lambda i,j: Lprod("Lko",i,j,2,r=r))

                return H(b[1],a[2])*G(a[1],b[2])-G(b[1],a[2])*H(a[1],b[2])+sympy.sympify(1)/(r-1)*((sum(
                    H(k,a[2])*H(-k,b[2]) for k in range(-r,r+1) if k!=0) if a[1]+b[1]==0 else 0)+(-sum(
                    H(b[1],k)*H(a[1],-k) for k in range(-r,r+1) if k!=0) if a[2]+b[2]==0 else 0))
            
        a1,b1,u,a2,b2,v=a[1:]+b[1:]
        R=(Lprod("Lk",a2,b1,v,r=r)*Lprod("Lk",a1,b2,u,r=r)-Lprod("Lk",a2,b1,u,r=r)*Lprod("Lk",a1,b2,v,r=r))*(1/(u-v))+(sum(Lprod("Lk",-k,b1,u,r=r)*Lprod("Lk",k,b2,v,r=r) for k in range(-r,r+1) if k!=0)*(1/(u-v+r-1)) if a1+a2==0 else 0)-(sum(Lprod("Lk",a2,-k,v,r=r)*Lprod("Lk",a1,k,u,r=r) for k in range(-r,r+1) if k!=0)*(1/(u-v+r-1)) if b1+b2==0 else 0)
        return R
    
    def expand(self,replace=True,inplace=False):
        """
        Expands each coefficient in terms and replaces L(u) with quadratic terms.
        
        Args:
            replace (bool): Whether to replace terms like L^(1)(u) with G, H.
            inplace (bool): Whether to perform the operation in place.

        Returns:
            Lprod: Expanded and possibly replaced instance.
        """
        if inplace:
            R=expand(self,replace=replace,inplace=False)
            self._terms=R._terms
            return self
        return sum(
                    Lprod({tuple([z  if z[0][-1]=="o" or z[0][0] in "GH" else ((z[0]+"o",)+z[1:-1]+(2-pows[n],) if not replace else ({1:"G",0:"H"}[pows[n]]+z[0][1:],)+z[1:3])
                                  for n,z in enumerate(i) if pows[n]!=2 or not replace]
                         ) : (fac.expand() if hasattr(fac,"expand") else fac)*sympy.Mul(*[i[k][-1]**pows[k]
                                             for k in range(len(i))])
                           },r=self.r)
            for i in self._terms for fac in [self._terms[i]]
                   for pows in itertools.product(*([range(3) if
                                                    z[0][-1]!="o" and z[0][0] not in "GH"
                                                    else
                                                    range(1) for z in i]))
                  if (not replace or all(i[k][1]==i[k][2]*(-1 if i[k][0][1]==i[k][0][2] else 1) for k in range(len(i)) if pows[k]==2 and i[k][0][-1]!="o" and i[k][0][0] not in "GH" ))
            )
    
    def simp(self,inplace=False):
        """
        Simplifies each coefficient of the terms.

        Args:
            inplace (bool): Whether to modify the instance in place.

        Returns:
            Lprod: Simplified instance.
        """
        if not inplace:
            self=self.copy()
        for i in self._terms:
            try: self._terms[i]=self._terms[i].simplify()
            except:pass
        for i in list(self._terms):
            if self._terms[i]==0: self._terms.pop(i)
        return self
    
    def replaceRect(self,a=None,s=None,r=None):
        """
        Replaces G components with a tensor object corresponding to the matrix representation.

        Args:
            a (int): Parameter representing the degree.
            s (int): Symbolic representation parameter.
            r (int): Index range.

        Returns:
            tensor: Tensor representation of the matrix.
        """
        assert not (a==None and s==None)
        a,s=[1 if x==None else x for x in (a,s)]
        if a==1: return self.replacesym(s,r)
        assert all(f[0][0]=="G" for t in self._terms for f in t)
        r,={r,self.r}.difference({None})

        if (a,s,r) not in Lprod.RectGs:
            G_=G(r=r, a=a, s=s)
            Lprod.RectGs[(a,s,r)]=tensor(numpy.array(G_.array))
        ThisG=Lprod.RectGs[(a,s,r)]
        self=self.into("ul",inplace=False)

        return sum(
            self._terms[i]*(functools.reduce(lambda x,y:x*y,
                [tensor(ThisG[r+i[k][1]-bool(i[k][1]>0),r+i[k][2]-bool(i[k][2]>0)])
                                            for k in range(len(i))])
                            if len(i) else 1+0*tensor(ThisG[0,0]))
            for i in self._terms)
    
    def replacesym(self,s,r=None):
        """
        Replaces Lprod object with a symbolic tensor matrix.

        Args:
            s (int): Symbolic representation parameter.
            r (int): Index range.

        Returns:
            tensor: Symbolic tensor matrix.
        """
        r,={r,self.r}.difference({None})

        if (s,r) in Lprod.symreps: Lo=Lprod.symreps[(s,r)]
        else:
            G_=G(r=r, a=1, s=s)
            g=tensor(numpy.array(G_.array))
            Lo=[ -((r-2)**2+2*(r-1)*s+s**2)/sympy.sympify(4)+(r-1)/sympy.sympify(2)*g+1/sympy.sympify(2)*(g*g),  #order 0
                g, # order 1
                1+0*g # order 2
            ]
            Lprod.symreps[(s,r)]=Lo

        self=self.into("ul",inplace=False)
        assert all(f[0][1:3]=="ul" for t in self._terms for f in t)

        return sum(
            self._terms[i]*sympy.Mul(*[i[k][-1]**pows[k]
                                             for k in range(len(i))])*(functools.reduce(
                                                     lambda x,y:x*y,
                [tensor(Lo[2-i[k][-1] if i[k][0][-1]=="o" and i[k][0][0]=="L"
                            else {"G":1,"H":0}.get(i[k][0][0],pows[k])][r+i[k][1]-bool(i[k][1]>0),r+i[k][2]-bool(i[k][2]>0)])##### to be relabelled !!!!!!!!
                        for k in range(len(i))])
                                                                       if len(i)>0 else
                                                                       1+0*tensor(Lo[1][0,0]))
            for i in self._terms for pows in itertools.product(*([range(3) if
                                                    z[0][-1]!="o" and z[0][0] not in "GH"
                                                    else
                                                    range(1) for z in i]))
        )
    
    def toswap(terms):
        """
        Determines pairs of terms that need to be swapped.

        Args:
            terms (list): List of terms to evaluate.

        Returns:
            list: List of indices of terms to swap.
        """
        return  [(i,k) for i in terms for k in range(len(i)-1)
            for sa,sb in [[0 if sum(x[1:3])==0 else (
                    -1 if max(x[1:3])<0 or (sum(x[1:3])>0 and min(x[1:3])<0) #anihilation
                    else 1) for x in (i[k],i[k+1])]]
            if sb>0>=sa or ((sa>0<sb or sb<=0>=sa) and
                            ((Lprod.isG(i[k]) and Lprod.isH(i[k+1]))
                             or (Lprod.isSame(i[k],i[k+1]) and i[k][1:3]>i[k][1:3] and (Lprod.isG(i[k]) or (i[k][1]+i[k+1][1]!=0 and i[k][2]+i[k+1][2]!=0)))))
                                ] #terms not sorted by creationness then by HG and finally by alphanumericity
    
    def isG(t):return Lprod.isGH(t,"G",1)
    def isH(t):return Lprod.isGH(t,"H",2)
    def isGH(t,l,o): return t[0][0]==l or (t[0][::3]=="Lo" and len(t)==4 and t[-1]==o)
    def isSame(t,u):[Lprod.isG(t),Lprod.isH(t)]==[Lprod.isG(u),Lprod.isH(u)]

    def type(*k):
        """
        Determines the type of the term.

        Args:
            k: Term(s) to evaluate.

        Returns:
            int: 1 for creation, -1 for annihilation, 0 for Cartan operator.
        """
        if len(k)>1 and isinstance(k[1],int): i,j=k[1:3]
        elif len(k)==1 and hasattr(k[0],"__iter__") and len(k[0])>=3: i,j=k[0][1:3]
        elif len(k)==1 and isisnstance(k[0],Lprod) and len(k[0]._terms)==1 and len(list(k[0]._terms)[0])==1: i,j=list(k[0]._terms)[0][0][1:3]
        elif len(k)==1 and isisnstance(k[0],Lprod) and len(k[0]._terms)==1 and len(list(k[0]._terms)[0])==0: return 0 # regard 1 as a Cartan generator
        else: raise ValueError("could not identify the generator")
        return -1 if i<0>j else (1 if i>0<j else (0 if i+j==0 else (1 if i+j<0 else -1)))
    
    def replace_right(self,inplace=True,right_replacements={},verbose=False,**args):
        """
        Replaces parts of terms based on a predefined mapping.

        Args:
            inplace (bool): Whether to modify the instance in place.
            right_replacements (dict): Mapping of replacements to perform.
            verbose (bool): Whether to print detailed logs.

        Returns:
            Lprod: Modified instance with replacements applied.
        """
        if inplace==False: return self.copy().replace_right(self,inplace=True,right_replacements=right_replacements,verbose=verbose,**args)
        if right_replacements=={}: return self

        while True:
            try:
                k,l=next((k,l) for k in self._terms for l in range(len(k)) if k[l:] in right_replacements)
                fac=self._terms.pop(k)
                self._terms=(self+fac*Lprod(k[:l],r=self.r)*right_replacements[k[l:]]).simp()._terms
            except StopIteration:
                break
        return self

    def sort(self,inplace=False,r=None,verbose=False,right_replacements={},**args):
        """
        Reorganizes terms into a "normal order" to streamline subsequent
        computations. While the reordering is not unique, it simplifies
        the process by efficiently calculating coordinates. The 
        right_replacements argument specifies how to reduce the tail of
         a product for greater clarity and ease of use. 

        Args:
            inplace (bool): Whether to modify the object in place.
            r (int): Rank for the operator.
            verbose (bool): Whether to print detailed logs.
            right_replacements (dict): Mapping of terms for replacement.

        Returns:
            Lprod: Sorted expression.
        """
        if inplace:
            R=Lprod.sort(self,r=r,inplace=False,verbose=verbose,right_replacements=right_replacements,**args)
            self._terms=R._terms
            return self
        
        r,={self.r,r}.difference({None})
        if r not in Lprod.sort.dict: Lprod.sort.dict[r]={}
        brgs=args

        # Step I sort by creation/cartan/annihilation
        self=self.into("k",inplace=False)
        while any(self._terms[k]==0 for k in self._terms):
            self._terms.pop(next(k for k in self._terms if self._terms[k]==0))

        if right_replacements: self.replace_right(inplace=True,right_replacements=right_replacements, verbose=verbose,**{k:brgs[k] for k in ["printleft"] if k in brgs})

        # sa and sb are: -1 for anihilation operators, 0 for cartan, and 1 for creation
        l=Lprod.toswap(self._terms)
        while l: #some terms not sorted
            i,k=l[0]
            b=self.copy()
            fac=b._terms.pop(i)
            if i[k:k+2] in Lprod.sort.dict[r]:
                c=Lprod.sort.dict[r][i[k:k+2]]
            else:
                coms=(Lprod(i[k],r=r).Lcom(i[k+1]),-Lprod(i[k+1],r=r).Lcom(i[k])) #The two possibilities for this commutator
                sw=[[] if c==0 else Lprod.toswap(c._terms) for c in coms]
                try:
                    c=next(c+Lprod(i[k:k+2][::-1],r=r) for n,c in enumerate(coms) if sw[n]==[])
                    Lprod.sort.dict[r][i[k:k+2]]=c
                except:
                    
                    if any(a[0][::3]=="Lo" for T in self._terms for a in T):
                        return self.expand(replace=True,inplace=False).sort(r=r,verbose=verbose,right_replacements=right_replacements,**args)
                    ### simplify some terms and list the others (the issue)
                    coms=[Lprod(i[k],r=r).Lcom(i[k+1],sort=True,rev=p) for p in [False,True]]
                    ### List the terms that will cause issues
                    LTerms=[l if i[k:k+2] in l else l+[i[k:k+2]]  for c in coms for l in [list(c.toswap._terms)]]
                    ComDicts=[{},{}]
                    while any(x not in ComDicts[k] for k in range(2) for x in LTerms[k]):
                        x=next(x for k in range(2) for x in LTerms[k] if x not in ComDicts[k])
                        cms=[Lprod.Lcom(*x,r=r,sort=True,rev=sgn) for sgn in [False,True]]
                        for _k in [k for k in range(2) if x in LTerms[k]]:
                            cm=[c for c in cms if set(c.toswap._terms).issubset(set(LTerms[_k]))]
                            if len(cm)==1: ComDicts[_k][x]=cm[0]
                            elif len(cm)==2: ComDicts[_k][x]=cm[_k]
                            else:
                                m=max(len(set(c.toswap._terms).intersection(set(LTerms[_k]))) for c in cms)
                                cm=next(c for c in cms if m==len(set(c.toswap._terms).intersection(set(LTerms[_k]))))
                                LTerms[_k]=list(set(LTerms[_k]).union(set(cm.toswap._terms)))
                                ComDicts[_k][x]=cm
                    
                    LMatrix=[
                     sympy.Matrix([[ComDicts[k][i]._terms.get(j,0) for j in LTerms[k]] for i in LTerms[k]])
                        for k in range(2) ]
                    LRHS=[[ComDicts[k][i].OK for i in LTerms[k]]  for k in range(2) ]
                    ## above we had comutators=Matrix.product+RHS... Changed below
                    LMatrix=[sympy.eye(l.shape[0])-l for l in LMatrix]
                    LRHS=[[LRHS[k][n]+Lprod(i[::-1],r=r) for n,i in enumerate(LTerms[k])]
                         for k in range(2) ]
                    ## now M.prods=RHS
                    inv=[None]*2
                    for _k in range(2):
                        try: inv[_k]=LMatrix[_k].inv()
                        except:
                            if verbose:
                                print(
                                      "  warning: matrix {} not invertible, in {}".format(LMatrix[_k],i[k:k+2]))
                    try: nc,=(n for n,i in enumerate(inv) if i!=None)
                    except:
                        if not any(inv):
                            ######## None of the two matrices is invertible :(
                            if i[k][0]=="Hll"==i[k+1][0] and i[k][1]+i[k+1][1]==0==i[k][2]+i[k+1][2]:
                                AllPairs=[(("Hll",i,j),("Hll",-i,-j)) for i in range(-r,r+1) for j in range(-r,r+1) if i*j]
                            else:
                                print(i[k:k+2])
                                Lprod.tmp=[i[k:k+2],LMatrix,LTerms]
                                raise ValueError
                            ######## Strategy: compute all possible relations that come to my mind, turn it into a huge matrix A (overdetermined system) and find a left inverse as (A.T*A).inv()*(A.T)
                            AllPairs=[(("Hll",i,j),("Hll",-i,-j)) for i in range(-r,r+1) for j in range(-r,r+1) if i*j]
                            LTerms=[k for k in AllPairs if Lprod.toswap([k])]
                            Matrix=[]
                            RHS=[]
                            for x in AllPairs:
                              for sgn in [True,False]:
                                y=Lprod.Lcom(*x,r=r,sort=True,rev=sgn,diff=True)
                                Matrix.append([-y._terms.get(l,0) for l in LTerms])
                                for _x in LTerms:
                                    if _x in y._terms: y._terms.pop(_x)
                                RHS.append(y)

                            M=sympy.Matrix(Matrix)
                            Inv=(M.T*M).inv()*(M.T)
                            nc,LRHS,inv,LTerms=0,[RHS],[Inv],[LTerms]

                        else:     
                            nc=1 if inv[0].shape[0]>inv[1].shape[0] else 0
                    ## At this point, at least one of them was inverted
                    ## hence fill the dict
                    slt=[sum(inv[nc][i,j]*LRHS[nc][j] for j in range(inv[nc].shape[1]))
                        for i in range(inv[nc].shape[0])]       #inv[nc]@LRHS[nc]
                    for nn,t in enumerate(LTerms[nc]):  Lprod.sort.dict[r][t]=slt[nn]
                    c=Lprod.sort.dict[r][i[k:k+2]]
            c=Lprod({i[:k]+t+i[k+2:]:c._terms[t] for t in c._terms},r=r)
            self=(fac*c+b).simp()
            if right_replacements: self.replace_right(inplace=True,right_replacements=right_replacements, verbose=verbose
                                                      )
            l=Lprod.toswap(self._terms)
        return self

    def cca(r):
        """
        Returns creation, Cartan, and annihilation operators.

        Args:
            r (int): Rank of the operators.

        Returns:
            tuple: (creation, Cartan, annihilation) lists.
        """
        cr,ca,an=[],[],[]
        for i in range(-r,r+1):
            for j in range(-r,r+1):
                if i*j==0: continue
                [(an if i<0>j else (cr if i>0<j else (ca if i+j==0 else (cr if i+j<0 else an)))
                  ).append(Lprod(l+"k",i,j,r=r)) for l in "GH"]
        return cr,ca,an
    
Lprod.__radd__=Lprod.__add__
Lprod.__rmul__=Lprod.__mul__ #if the second factor is not an Lprod then mul is commutative
# Lprod.GH=True #use notations G and H instead of L⁽¹⁾ and L⁽²⁾
Lprod.symreps={}
Lprod.RectGs={}

class Lvect:
    """ 
    The Lvect class introduces a higher-level structure in the
    codebase, combining the concept of a Lprod with a "vacuum" state
    and additional properties like weights. It is designed for 
    applications in algebraic computations, particularly involving 
    operators in the context of theoretical physics or advanced algebraic structures. """
    def __init__(self,Lp,**args):
        """
        Args:
            Lp (Lprod): An instance of `Lprod` or a valid input to create one.
            **args: Additional parameters like 'r' (rank) or 'weights'.
        """
        self.Lprod=Lp.copy() if isinstance(Lp,Lprod) else Lprod(Lp,r=args["r"]).copy()
        self.r,={self.Lprod.r,args.get("r",None)}.difference({None})
        if self.Lprod.r==None: self.Lprod.r=self.r
        self.weights=args.get("weights",None)
        if not args.get("nosimp",False):
            self.simp(inplace=True)

    def __eq__(self,fles):
        if not isinstance(fles, Lvect):
            if fles==0: return all(self.Lprod._terms[x]==0 for x in self.Lprod._terms)
            return False
        if self.r!=fles.r: return False
        if self.weights!=fles.weights: return False
        return self.Lprod==fles.Lprod
    
    def __hash__(self): return (self.r,tuple(self.weights),self.Lprod).__hash__()

    def vac(**args): return Lvect({():1},**args)

    def __repr__(self): return self.Lprod.__repr__("|0>")

    def __add__(self,B):
        if B==0: return self.copy()
        if isinstance(B,Lvect):
            st={x.r for x in (self,B)}.difference({None})
            r,=(None,) if len(st)==0 else st
            weights=None if B.weights==None==self.weights else B.weights if B.weights==self.weights or None in [B.weights,self.weights] else 1/0
            return Lvect(B.Lprod+self.Lprod,r=r,weights=weights,nosimp=True)
        raise ValueError
    
    def __rmul__(self,L: Lprod):
        res=self.copy()
        res.Lprod=L*res.Lprod
        return res
    
    def copy(self):
        return Lvect(self.Lprod.copy(),r=self.r,weights=self.weights,nosimp=True)
    
    def simp(self, inplace=True,**args):
        """
        Simplifies the `Lvect` object by:
        - Sorting terms in `Lprod`.
        - Replacing terms based on weights.
        - Removing annihilation operators.

        Args:
            inplace (bool): Whether to modify the object in place.

        Returns:
            Lvect: Simplified `Lvect` object.
        """
        if not inplace: return self.copy().simp(inplace=True,**args)

        args.get("verbose",False) and print(next(l[:80]+("..." if len(l)>80 else "") for l in
                [args.get("printleft","")+"simp("+str(self)+r","+",".join(k+"="+str(args[k]) for k in args)]))
        
        if "printleft" in args:
            for c,d in [("├","│"),("─"," "),("└"," ")]:
                args["printleft"]=str.replace(args["printleft"],c,d)
        brgs=args.copy()
        brgs["printleft"]=args.get("printleft","")+" ├─"
        self.Lprod.sort(inplace=True,**{k:{y:args[l][x].Lprod for x in args[l] for y in x.Lprod._terms} for (k,l) in [("right_replacements","replacements")] if l in args},**{k:brgs[k] for k in ["verbose","printleft"] if k in brgs})
        w=self.weights
        u=sympy.Symbol("u")
        r=self.r

        if w==None: w=Rectweights(r) # use (2.19) to replace the λᵢ(-u) of (2.12)
        for i in list(self.Lprod._terms):
            if len(i)>0 and Lprod.type(i[-1])==-1:
                self.Lprod._terms.pop(i)
                continue
            i0,fac=i,self.Lprod._terms[i]
            while len(i)>0 and Lprod.type(i[-1])==0:
                a=i[-1][2]
                _w=w[r+a-(1 if a>0 else 0)]
                if i[-1][0][0]=="L" and i[-1][0][-1]!="o":
                    _w=_w.replace(u,i[-1][-1])
                elif i[-1][0][0]=="G" or (i[-1][0][0]=="L" and i[-1][-1]==1):
                    _w=_w.coeff(u,1)
                elif i[-1][0][0] in "HL":
                    _w=_w.coeff(u,0)
                fac=fac*_w
                i=i[:-1]
            self.Lprod._terms.pop(i0)
            self.Lprod._terms[i]=self.Lprod._terms.get(i,0)+fac
        self.Lprod.simp(inplace=True)#,**brgs)
        args.get("verbose",False) and print(next(l[:80]+("..." if len(l)>80 else "") for l in
                [args.get("printleft","")+" └─"+str(self)]))
        return self
    
    def terms(self):
        """Break the Lprod into individual terms, each wrapped in an Lvect"""
        return [Lvect(Lprod({t:self.Lprod._terms[t]},r=self.r),weights=self.weights,nosimp=True)
            for t in self.Lprod._terms]
    
    def coordinates(self,**args):
        """If neither Lprod nor Gcoords is requested, return coordinates in a simplified form"""
        if not args.get("Lprod",False) and not "Gcoords" in args:
            return next({} if x==0 else x._terms for x in [Lvect.coordinates(self,**args,Lprod=True)])
        
        # Handle Gcoords argument to extract specific coordinates
        if "Gcoords" in args:
            (a,r,s)=args["Gcoords"]
            coo=self.replaceRect(a=a,s=s)
            coo={n:i for n,i in enumerate(coo) if i!=0}
            # If basis and replacements are specified, update them
            if "basis" in args and "replacements" in args and not ("nosave" in args) and len(self.Lprod._terms)==1:
                update_replacement_and_basis(vect=self,coord=coo,**args)
            return coo
        
        # Use rank `r` for further calculations
        r=self.r
        if len(self.Lprod._terms)==0: return Lprod(0,r=r)

        # Cache commonly used groups of operators: creation, cartan, annihilation
        cr,ca,an=args.get("cca",Lprod.cca(r))
        args["cca"]=cr,ca,an

        # Retrieve the dictionary of coordinates for this rank and weight combination
        try: dct=args.get("dct",next(l[2] for l in Lvect.coordinates.remember if l[0]==r and l[1]==self.weights))
        except StopIteration:
            # Initialize a new dictionary if not found in the cache
            Lvect.coordinates.remember.append([r,self.weights,{tuple():Lprod(1,r=r)}])
            dct=Lvect.coordinates.remember[-1][-1]

        ### the program has to do a tree traversal to explore all relevent combinations of anihilation operators. Hence it is programmed as a recursive algorithm.
        args["dct"]=dct
        try:
            t,=self.Lprod._terms
        except ValueError: pass
        else:
            f=self.Lprod._terms[t]
            try: return f*dct[t]
            except KeyError: pass
        
        args.get("verbose",False) and print(next(l[:80]+("..." if len(l)>80 else "") for l in
    [args.get("printleft","")+"coordinates("+str(self)+r","+",".join(k+"="+str(args[k]) for k in args)]))
        
        if "printleft" in args:
            for c,d in [("├","│"),("─"," "),("└"," ")]:
                args["printleft"]=str.replace(args["printleft"],c,d)

        brgs=args.copy()
        brgs["printleft"]=args.get("printleft","")+" ├─"

        if "basis" in args and "replacements" in args and not "nosave" in args and len(self.Lprod._terms)==1:
            coord=self.coordinates(nosave=True,**brgs) #brgs contains Lprod=True
            update_replacement_and_basis(vect=self,coord=coord,**brgs)
            
        s0=self
        ### start by simping it and using replacements if available
        self=self.simp(inplace=False,**{k:brgs[k] for k in ["replacements","verbose","printleft"] if k in brgs})
        if "nosave" in args: [args.pop("nosave"),brgs.pop("nosave")]
        
        if len(self.Lprod._terms)==0:
            if len(s0.Lprod._terms)==1:
                dct[next(x for x in s0.Lprod._terms)]=Lprod(0,r=r)
                if "basis" in args and "replacements" in args and not ("nosave" in args) :

                    update_replacement_and_basis(vect=self,coord=Lprod(0,r=r),**brgs)
            return Lprod(0,r=r)
        
        if len(self.Lprod._terms)>1:
            return sum(L.coordinates(**brgs) for L in self.terms()) #brgs contains Lprod=True
        
        t,=self.Lprod._terms
        f=self.Lprod._terms[t]
        T=self.copy();T.Lprod._terms[t]=1

        if t in dct: return f*dct[t]

        if t==(('Gll',1,2),) :
               Lvect.tmp=[t,T,f,[(a,cb,cba)  for a in an
                for vb in [(a*T).simp(**{k:brgs[k] for k in ["replacements","verbose","printleft"] if k in brgs})]
                for cb in [vb.coordinates(**brgs)] for cba in [cb*a]  #brgs contains Lprod=True
                if len(cba._terms)!=0]]
               print("\n"*30+" That's here "+"\n"*30)
               return 5
        
        res=sum(cba
                for a in an
                for vb in [(a*T).simp(**{k:brgs[k] for k in ["replacements","verbose","printleft"] if k in brgs})]
                for cb in [vb.coordinates(**brgs)] for cba in [cb*a]  #brgs contains Lprod=True
                if len(cba._terms)!=0)
       
        if res==0: res=Lprod(0,r=r)
        dct[t]=res
        
        ###### Add a check of what goes wrong
        for x in res._terms:
            z=(Lprod(x)*Lvect(t,r=self.r,weights=self.weights)).simp(**{k:brgs[k] for k in ["replacements"] if k in brgs})
            if not z==res._terms[x]*Lvect.vac(r=self.r,weights=self.weights):
                print("\n\n\n\n\nWarning: error in component {} of the coordinate of {}, gives {} vs {}\n\n\n\n\n".format(Lprod(x),Lvect(t,r=self.r,weights=self.weights),z,res._terms[x]))

        if t==(('Gll',1,2),) :
               Lvect.tmp.append(res);Lvect.tmp.append(res.copy())
               raise ValueError
        
        if "basis" in args and "replacements" in args and not ("nosave" in args) :
            update_replacement_and_basis(**brgs)
        return f*res
    
    def replacesym(self,s):
        """
        Replaces the `Lprod` in the current `Lvect` with its matrix representation
        in the symmetric representation corresponding to the weight `s`.
        """
        r=self.r
        # Replace the `Lprod` with the corresponding matrix in the symmetric representation
        O=self.Lprod.replacesym(r=r,s=s)
        # Ensure the highest weight state for (r, s) exists in `symweights.dict`    
        if (r,s) not in symweights.dict: symweights(r=r,s=s)
         # Retrieve the highest weight state index
        hws=symweights.dict[(r,s)][1]
        # Return the specific column corresponding to the highest weight state
        return O[:,hws]
    
    def replaceRect(self,a=None,s=None):
        """
        Replaces the `Lvect` with its rectangular representation for given `a` and `s`.
        If `a == 1`, it uses the symmetric representation (`replacesym`).
        Otherwise, it calculates the rectangular weights and determines the highest weight state.
        """
        # Ensure either `a` or `s` is provided
        assert not (a==None and s==None)
        # Default to `a=1` or `s=1` if not provided
        a,s=[1 if x==None else x for x in (a,s)]

        # If `a == 1`, use the symmetric representation
        if a==1: return self.replacesym(s)

        r=self.r
        # Replace the `Lprod` with its rectangular matrix representation
        O=self.Lprod.replaceRect(a=a,s=s,r=r)

        # Check if the rectangular weights for (a, r, s) are already in `symweights.dict`
        if (a,r,s) not in symweights.dict:
            # Construct annihilation operators for the given `a` and `s`
            anihil=[Lprod("Gk",a_,b_).replaceRect(a=a,s=s,r=r)
                for a_ in range(-r,r+1) for b_ in range(-r,r+1) if a_*b_!=0
                 and (min(a_,b_)<0<a_+b_ or 
                      max(a_,b_)<0)]
            # Determine the highest weight state index
            hws,=[i for i in range(anihil[0].array.shape[-1]) if all((k[:,i]==0).all() for k in anihil)]
            # Store rectangular weights and the highest weight state in `symweights.dict`
            symweights.dict[(a,r,s)]=(Rectweights(r,a=a,s=s),hws)

        # Retrieve the highest weight state index
        hws=symweights.dict[(a,r,s)][1]
        # Return the specific column corresponding to the highest weight state
        return O[:,hws]
        
Lvect.coordinates.remember=[] # a list of triples r,weights,dict
Lvect.__radd__=Lvect.__add__

def symweights(s,r,**args):
    r"""
    Computes the weight functions λᵢ(-u) in the symmetric representation s.
    
    Args:
    * s (int): Symmetric representation index.
    * r (int): Rank.
    * args (dict): Optional arguments, including:
        - Check214 (bool): If True, verifies formula (2.12).

    Returns:
    * List of weight functions λᵢ(-u) for the given representation.
    """
    if (r,s) not in symweights.dict:
        # Generate annihilation operators
        anihil=[Lprod(L,a,b,*o).replacesym(s=s,r=r) for L,*o in [("Lko",2),("Gk",)]
                for a in range(-r,r+1) for b in range(-r,r+1) if a*b!=0
                 and (b>-a>0 or 
                      max(a,b)<0)]
        
        # Identify the highest weight state (hws) column
        hws=next(i for i in range(anihil[0].array.shape[-1]) if all((k[:,i]==0).all() for k in anihil))

        # Optional: Check formula (2.12) if 'check214' argument is set
        if args.get("check214",False):
            # First condition: 0 < a < -b
            anihil=[Lprod(L,a,b,*o).replacesym(s=s,r=r) for L,*o in [("Lko",2),("Gk",)]
                for a in range(-r,r+1) for b in range(-r,r+1) if a*b!=0
                and 0<a<-b]
            if not all((k[:,hws]==0).all() for k in anihil): print("(2.14) does not hold")

            # Second condition: a > -b > 0
            anihil=[Lprod(L,a,b,*o).replacesym(s=s,r=r) for L,*o in [("Lko",2),("Gk",)]
                for a in range(-r,r+1) for b in range(-r,r+1) if a*b!=0
                    and a>-b>0]
            if all((k[:,hws]==0).all() for k in anihil): print("(2.14) would hold if 'i>j' was replaced by 'i<j'")

        # Compute the λᵢ(-u) values and store in the cache
        symweights.dict[(r,s)]=([Lprod("Lk",-a,a,sympy.Symbol("u")).replacesym(s=s,r=r)[hws,hws] for a in range(-r,r+1) if a!=0],hws)
    return symweights.dict[(r,s)][0] # Return the λᵢ(-u) values


def Rectweights(r,a=None,s=None,**args):
    """
    Computes the weights in rectangular representations based on parameters `r`, `a`, and `s`.

    Args:
    * r (int): Rank.
    * a (int, optional): Rectangular representation index. Defaults to 1.
    * s (int, optional): Symmetric representation index. Defaults to 1.
    * args (dict): Additional options, including:
        - Full (bool): If True, includes all terms in the computation.

    Returns:
    * List of weight functions for the given representation.
    """
    u=sympy.Symbol("u")
    
    # If both `a` and `s` are None, return default weights based on formula (2.19)
    if a==None==s:
        return [u**2+u*sympy.Symbol(r"\lambda_{%s}^{[1]}"%a)+sympy.Symbol(r"\lambda_{%s}^{[2]}"%a) for a in range(-r,r+1) if a!=0]
    
    # Default `a` and `s` to 1 if not provided
    a,s=(1 if x==None else x for x in [a,s])
    assert a<=r # Ensure `a` is within valid range

    # Compute weights for rectangular representations
    return [u**2+u*(s if -a<=k<=0 else (-s if 0<=k<=a else 0))+(
        (-((-a+r-s-1)*(-a+r+s-1)/sympy.sympify(4) if abs(k)<=a else (-a+r+s-1)**2/sympy.sympify(4))
        if args.get("Full",True) else
        sympy.Symbol(r"\lambda_{%s}^{[2]}"%k)))
            for k in range(-r,r+1) if k!=0]

# Initialize dictionaries for caching
Lprod.sort.dict={}
symweights.dict={}

# ## Global Remarks about so Yangian
# * $[L_{a,b}(u),L_{a,b}(v)]=0$ because when we expand this commutator, it is independent of $u^and $v$, and has to be zero at $u=v$, hence it is always zero
# * I think there was a mistake with the def of HWS in equation (2.14) as compared to (2.12)






############ Sanity Checks ################

def update_replacement_and_basis(basis,replacements,vect=None,coord=None,dct=None,**args):
    """
    Update the dictionaries `basis` and `replacements` using either:
    - The coordinate `coord` of an `Lvect` `vect`, or
    - The dictionary `dct` of coordinates.

    Args:
    - basis (dict): Dictionary to store basis vectors.
    - replacements (dict): Dictionary to store replacement rules.
    - vect (Lvect, optional): Vector for which coordinates are being computed. Default is None.
    - coord (dict, optional): Coordinate of the vector. Default is None.
    - dct (dict, optional): Dictionary of coordinates. Default is None.
    - args (dict): Additional arguments, including verbosity.
    
    Returns:
    - None: Updates `basis` and `replacements` in place.
    """
    # If no vector provided, process the dictionary `dct`
    if vect==None:
        # if not Yangian_irrep.changed: return None
        for v_ in dct:
            # Convert tuple `v_` into `Lvect` object
            v0=list(replacements)[0]
            v=Lvect(v_,r=v0.r,weights=v0.weights)

            # Skip if vector `v` is already handled
            if v not in Yangian_irrep.kept_free and v not in basis and v not in replacements and v.Lprod._terms!={}:
                # Recursively process new vectors
                update_replacement_and_basis(basis=basis,replacements=replacements,vect=v,coord=dct[v_],**args)
        # Yangian_irrep.changed=False
        return None
    
    # Processing specific vector `vect` and its coordinates `coord`
    nv=vect # Normalized vector
    nc=coord # Normalized coordinate

    # Skip processing if vector is in the "kept free" list
    if nv in Yangian_irrep.kept_free: return None

    # Skip processing if vector has multiple terms
    if len(vect.Lprod._terms)!=1: return None

    # Extract the single term and its coefficient
    t,=vect.Lprod._terms
    if vect.Lprod._terms[t]!=1:
        # Normalize coordinate by dividing the coefficient
        nc=1/sympy.sympify(vect.Lprod._terms[t])*nc
        nv.Lprod._terms[t]=1

    # Check if vector is already represented in `basis`, `replacements`, or `kept_free`
    if any(set(v.Lprod._terms)=={t} and v.Lprod._terms[t]==1
            for d in [replacements,basis,Yangian_irrep.kept_free] for v in d):
        return None
    
    # Normalize the coordinate if it's an `Lvect` or `Lprod`
    if isinstance(nc,Lvect):nc=nc.Lprod
    if isinstance(nc,Lprod):nc=nc._terms

    # Handle the case of zero coordinates
    if nc=={}:
        args.get("verbose",False) and print(args.get("printleft","")+"  ",nv,"=0")
        replacements[nv]=0*nv
        savetmp()
        return None
    
    # Compute matrix of coordinates for linear independence check
    components=functools.reduce(lambda a,b:a.union(b),[set(basis[b]) for b in basis]+[set(nc)])
    lb=list(basis)
    MatCoord=sympy.Matrix([[coords.get(c,0) for c in components] for coords in [basis[b] for b in lb]+[nc]])
    args.get("verbose",False) and print(args.get("printleft","")+"  computed coordinates matrix for ",nv)

    # Compute determinant to check linear independence
    dt=(MatCoord*MatCoord.T).det()
    if dt==0:
        # Handle dependent case: insert a rule in `replacements`
        try:
            coefs=MatCoord[:-1,:].T.solve(MatCoord[-1,:].T)
            Lprod.tmp=(coefs,lb,nv)
            replacements[nv]=sum(coefs[n]*b for n,b in enumerate(lb) if coefs[n]!=0)
            if replacements[nv]==0: replacements[nv]=0*nv
            if basis is Yangian_irrep.basis and replacements is Yangian_irrep.replacements: 
                savetmp()
        except: print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n BIG WARNING \n\n\n\n\n\n\n\n\n\n\n\n\n\n Non invertible matrix for Basis \n\n\n\n\n\n\n")
        return None
    
    # Handle independent case: add vector to `basis`
    if len({type(y) for x in basis for y in basis[x]})==1:
        basis[nv]=nc
        Yangian_irrep.kept_free=[]
        args.get("verbose",False) and print(args.get("printleft","")+"  linearly independent\n"+args.get("printleft","")+"  ->  ",list(basis))
        args.get("verbose",False) and print(args.get("printleft","")+"  flushed  Yangian_irrep.kept_free")
        if basis is Yangian_irrep.basis and replacements is Yangian_irrep.replacements: 
            savetmp()
    else: Yangian_irrep.kept_free.append(nv)
    return None

def Yangian_irrep(r,a=None,s=None,**args):
    """
    Constructs the Yangian irreducible representation for given parameters.
    
    Args:
    - r (int): The Yangian parameter for the algebra.
    - a (int): Parameter related to the highest weight.
    - s (int): Symmetry representation parameter.
    - args (dict): Additional options like "resume" or "verbose".

    Returns:
    - basis (dict): The basis of the Yangian representation.
    """
    assert not a==None==s

    weights=Rectweights(r=r,a=a,s=s) # Compute the weights for the given parameters.
    Cr,Ca,An=Lprod.cca(r) # Separate creation, Cartan, and annihilation operators.


    # Separate creation operators into `G` and `H` types.
    GCr,HCr=([i for i in Cr if next(t[0] for t in i._terms)[0][0]==l] for l in "GH")
    LV=Lvect.vac(r=r,weights=weights) # Initialize the vacuum vector.

    # Initialize or resume the basis and replacements dictionaries.
    if not args.get("resume", False):
        Yangian_irrep.basis={}
        Yangian_irrep.replacements={}

    basis=Yangian_irrep.basis
    if not args.get("resume", False):
        # Add vacuum coordinates to the basis.
        basis[LV]=LV.coordinates(Gcoords=(a,r,s),**args)
        
        # Add replacements for annihilation operators.
        for an in An:
            x=LV.copy()
            x.Lprod=an
            Yangian_irrep.replacements[x]=0*LV

        # Add replacements for Cartan operators.
        for ca in Ca:
            Yangian_irrep.replacements[ca*LV]=(ca*LV).simp()
    replacements=Yangian_irrep.replacements #set of rules when it understands that vectors are not linearly independent
    # We should turn my tree travel into a breath-first tree traval

    # Loop through creation operators: `G` in the first pass, `H` in the second.
    for n_Crs,Crs in enumerate([GCr,GCr+HCr]):
        if n_Crs==1:
            for nv in list(basis):
                if list(nv.Lprod._terms)==[(("Gll",1,3),)]: return None### this is a debuging attempt
                if any(type(k)==int for k in basis[nv]): #ie this one has its old coordinates
                    args.get("verbose",False) and print("computing coordinates of ",nv)
                
                    start_time = time.time()
                    nc=nv.coordinates(replacements=replacements,basis=basis,**args)
                    args.get("verbose",False) and print("done in %s s"% round(time.time() - start_time,4))
                    basis[nv]=nc
                    Yangian_irrep.kept_free=[]
                    savetmp()

        # Perform breadth-first traversal to build the basis.
        n=0
        while any(len(x)==n for v in basis for x in v.Lprod._terms):
            # Gather all vectors of the current length `n`.
            ll=[v for v in basis for x in v.Lprod._terms if len(x)==n] #list of leaves from which to continue
            for leaf in ll:
                # Apply each creation operator to the current leaf.
                for creation in Crs:
                    nv=creation*leaf
                    # Skip if the vector already exists in the basis or replacements.
                    if any(k.Lprod._terms==nv.Lprod._terms for k in list(basis)+list(replacements)): continue #because we are looping twice with different sets of creation operators

                    try:
                        if list(nv.Lprod._terms)[0] in next(rb[-1]for rb in Lvect.coordinates.remember if rb[:2]==[r,weights]): already=True
                        else: already=False
                    except StopIteration: already=False

                    # Compute the new vector's coordinates.
                    start_time = time.time()
                    (not already) and args.get("verbose",False) and print("computing coordinates of ",nv)
                    nc=nv.coordinates(replacements=replacements,basis=basis,**args,**{k:(a,r,s) for k in ["Gcoords"] if n_Crs==0})
                    (not already) and args.get("verbose",False) and print(nv," done in %s s"% round(time.time() - start_time,4))

                    # Update replacements and basis.
                    if nv in basis or nv in replacements: continue
                    else:
                        print("warning about replacements and basis")
                        update_replacement_and_basis(basis=basis,replacements=replacements,vect=nv,coord=nc,**args)

                        # Check consistency and update again if needed.
                        if len({type(y) for x in basis for y in basis[x]})==1:
                            update_replacement_and_basis(basis=basis,replacements=replacements,dct=next(rb[-1]for rb in Lvect.coordinates.remember if rb[:2]==[r,weights]),**args)
                            update_replacement_and_basis(basis=basis,replacements=replacements,vect=nv,coord=nc,**args)
                            assert nv in basis or nv in replacements
                        #Check that update_replacement_and_basis was run
            n=n+1 # Move to the next level.
    return basis

# Initialize global dictionaries for the Yangian representation.
Yangian_irrep.basis={}
Yangian_irrep.kept_free=[]
Yangian_irrep.replacements={}


def savetmp():
    """
    Saves the current state of the Yangian irreducible representation, including basis
    and replacement mappings, to a temporary Python file (.tmp.py) for recovery or debugging.
    """
    try: 
        # Attempt to find an existing .tmp.py file in the current directory.
        fn=next(os.getcwd()+"/"+x for x in os.listdir() if x.endswith(".tmp.py"))
    except StopIteration: 
        # If no file exists, create a new temporary file with a unique name.
        fn=bash("mktemp -q "+os.getcwd()+"/XXXXXX.tmp.py").rstrip('\n')

    # Extract the weights from the Yangian basis (assumes at least one element in the basis).
    w=next(v.weights for v in Yangian_irrep.basis)
    # Format weights as a list of sympy expressions in string format.
    w="["+",".join('coder.sympy.sympify("{}")'.format(i) for i in w)+"]"

    # Helper function to convert a dictionary to a string representation for saving.
    def sd(d): return "{"+",".join(str(k)+':coder.sympy.sympify("'+str(d[k])+'")' for k in d)+"}"

    # Start building the save string.
    # Define a lambda function `Lv` to recreate Lvect objects.
    s='Lv=lambda x:coder.Lvect(x,r={},weights={},nosimp=True)'.format(
        next(v.r for v in Yangian_irrep.basis),w)
    
    # Add the Yangian_irrep.basis data to the save string.
    s=s+'\n'+"coder.Yangian_irrep.basis={"+",\n ".join(
        "Lv("+str(next(t for t in v.Lprod._terms))+"):"+sd(Yangian_irrep.basis[v])
            for v in Yangian_irrep.basis)+"\n}\n"
    
    # Add the Yangian_irrep.replacements data to the save string.
    s=s+'\n'+"coder.Yangian_irrep.replacements={"+",\n ".join(
        "Lv("+str(next(t for t in v.Lprod._terms))+"):Lv(coder.Lprod("+sd(Yangian_irrep.replacements[v].Lprod._terms)+"))"
            for v in Yangian_irrep.replacements)+"\n}"
    
    # Write the constructed save string to the temporary file.
    with open(fn,"w") as w:w.write(s)

def savelog(xx): 
    """
    Saves the provided log data (`xx`) to a temporary log file (.tmp.log).
    
    Args:
    - xx: The log data to save, which will be converted to a string format.
    """
    try: 
        # Attempt to find an existing .tmp.log file in the current directory.
        fn=next(os.getcwd()+"/"+x for x in os.listdir() if x.endswith(".tmp.log"))
    except StopIteration: 
        # If no file exists, create a new temporary log file.
        fn=bash("mktemp -q "+os.getcwd()+"/XXXXXX.tmp.log").rstrip('\n')

    # Convert the log data to a string format.
    s=str(xx)
    
    # Write the log data to the temporary log file.
    with open(fn,"w") as w:w.write(s)