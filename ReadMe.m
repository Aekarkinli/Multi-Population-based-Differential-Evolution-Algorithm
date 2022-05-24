%{

% Analitic model for circle; (x-x0).^ + (y-y0).^2=r^2


Circle Fitting to hypothetical noise point cloud by using MDE, please run the code given below.



%}

rng(100);
r=10;
t=linspace(-pi,pi,5000)';
x=sin(t);
y=cos(t);
noisex=0.05*(rand(size(x,1),1)-0.50);
noisey=0.05*(rand(size(x,1),1)-0.50);
mydata.x=r*(x+noisex);
mydata.y=r*(y+noisey);
plot(mydata.x, mydata.y,'.r','markersize',1); shg, daspect([1 1 1]); hold on
%   algo_MDE(objfun      , mydata ,   N , D , low  , up  , MaxCycle , seed )
out=algo_MDE('fitCircle' , mydata ,  30 , 3 , -100 , 100 , 1000     , 100);
disp('Computed values for x0,y0 and radii,r ;')
[out,x0,y0,r]=fitCircle(out.bestsol,mydata)
x=r*(x-x0); y=r*(y-y0);
plot(x,y,'-b','linewidth',1), shg
axis tight