F77=gfortran
F2PY=f2py
#FFLAGS="-fPIC"
FFLAGS="-fPIC -fcheck=bounds"

npfun.pyf: npfun.f Makefile
	$(F2PY) --overwrite-signature -m npfun -h npfun.pyf only: fnp : npfun.f 
	$(F2PY) -DF2PY_REPORT_ON_ARRAY_COPY=1 --f77flags=$(FFLAGS) -c npfun.pyf -lgfortran npfun.f
clean:
	rm -f *.pyf
	rm -f *.o
	rm -f *.so
