      subroutine fnp(u,Fn,Dn,Mkpq,links,clinks,N,res)
Cf2py double complex dimension(N,3,2), intent(in) :: u
Cf2py double precision dimension(N,2), intent(in) :: Dn
Cf2py double complex dimension(N,3,2), intent(in) :: Fn
Cf2py double precision dimension(N,3,3,3,2), intent(in) :: Mkpq
Cf2py integer dimension(N,15,2), intent(in) :: links
Cf2py integer dimension(N,15,2), intent(in) :: clinks
Cf2py integer,optional,depend(u), intent(in) :: N=shape(u,0)
Cf2py double complex dimension(N,3,2), depend(N) ,intent(out) :: res
      implicit none
      double complex u(N,3,2),Fn(N,3,2),res(N,3,2),nl(2),i,up,upp,bp,bpp
      double precision Dn(N,2),Mkpq(N,3,3,3,2)
      integer N,links(N,15,2),clinks(N,15,2),j,l,m,k,ls
      parameter (i=dcmplx(0,1))
      do l=1,N
         do j=1,3
            nl(1)=0
            nl(2)=0
            do ls=1,15
               if (links(l,ls,1).ge.0) then
                  do m=1,3
                     do k=1,3
                        up=u(links(l,ls,1),k,1)
                        upp=u(links(l,ls,2),m,1)
                        bp=u(links(l,ls,1),k,2)
                        bpp=u(links(l,ls,2),m,2)
                        if(clinks(l,ls,1).gt.0) then
                           up=conjg(up)
                           bp=conjg(bp)
                        endif
                        if(clinks(l,ls,2).gt.0) then
                           upp=conjg(upp)
                           bpp=conjg(bpp)
                        endif
                        nl(1)=nl(1)-i*Mkpq(l,j,m,k,1)*(up*upp-bp*bpp)
                        nl(2)=nl(2)-i*Mkpq(l,j,m,k,2)*(up*bpp-bp*upp)
                     enddo
                  enddo
               endif
            enddo
            res(l,j,1)=-Dn(l,1)*u(l,j,1)+Fn(l,j,1)+nl(1)
            res(l,j,2)=-Dn(l,2)*u(l,j,2)+Fn(l,j,2)+nl(2)
         enddo
      enddo
      return
      end
