%{

AUTHOR: AHMET EMIN KARKINLI
MULTI-POPULATION BASED DIFFERENTIAL EVOLUTION ALGORITHM

See for ReadMe.m for example usage

%}

function out  =  algo_MDE(objfun , mydata , N , D , low , up , MaxCycle , seed )
rng(seed) ;
t = 3;
if size(low , 1)  ==  1 ,  low  =  repmat( low ,  [1 , D] ) ; up  =  repmat( up ,  [1 , D] ) ; end
% initialization of patterns and their objective function values
A  =  rand( t*N ,  D )  .*  (up - low ) + low ;
fitA  =  feval( objfun ,  A ,  mydata ) ; % see Eq.1-2
% Update # Initialization
[~ , ind] = min(fitA);
Best = A(ind , :);
% Iterative Search Phase
temp = A(1:N , :);      % pre-memory
noise = 0*A(1:N , :);  % initial value
for cycle  =  1 : MaxCycle

    j0 = randperm(t*N , N);
    B = A(j0 , :);
    fitB = fitA(j0);

    % mutation phase
    for i = 1:N
        scale  =  abs(randi([0 1])-rand^randi(10)) * randn.^randi(5);
        while 1 ,  r = randperm(N , 2); if r~= i ,  break; end ,  end
        for j = 1:D
            if rand  <  0.50 ,  dx = B(r(1) , j); else ,  dx = Best(j); end
            if rand  <  0.50 ,  dy = B(i , j);    else ,  dy = B(r(2) , j); end
            temp(i , j)  = B(r(2) , j) + scale  .*  ( dx - dy ) ;
        end
    end
    if rand^randi(5)  <  0.50 ,  c = 1; else ,  c = D; end
    map  =  abs( randi([0 1] , N , c) - rand(N , D).^randi(5 , N , c) )  <  0.50 ;
    % crossover phase
    % temp  =  map .*  B + ~map .* temp + ( ~map .*  moment );
    Trial  =  B + map .* (temp + noise- B);
    % border control
    Trial = borderControl(Trial , low , up);
    % update #1
    fittemp = feval( objfun ,  Trial ,  mydata ) ;
    ind = fittemp < fitB;
    fitB(ind) = fittemp(ind);
    B(ind , :) = Trial(ind , :);
    % update #2
    A(j0 , :) = B;
    fitA(j0) = fitB;
    % update the global solution
    [BestVal , ind] = min(fitA);
    Best = A(ind , :);
    % Report the iteration results to workspace and screen
    out.bestval  =  BestVal ;
    out.bestsol  =  Best ;
    % assignin('base' ,  'solution_DEv2' ,  out )
    fprintf('%5.0f -- >  %5.16f \n' ,  cycle ,  BestVal ) ;
    noise  =  B  .*  10.^randi([-12 -9] , N , D) .* (rand(N , 1)-0.50);
end

% Border Control
function x = borderControl(x , low , up)
[N , D] = size(x);
for i = 1:N
    for j = 1:D
        if x(i , j) < low(j) ,  x(i , j) = min(rand*low(j) , low(j)); end
        if x(i , j) > up(j) ,   x(i , j) = max(rand*up(j) , up(j)); end
    end
end