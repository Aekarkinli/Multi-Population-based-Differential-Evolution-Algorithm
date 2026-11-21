% Runs MATLAB MDE on the benchmark set and writes matlab_results.csv
names = {'sphere','rastrigin','rosenbrock','ackley','shifted_sphere'};
lows  = [-100, -5.12, -30, -32, 1];
ups   = [ 100,  5.12,  30,  32, 5];
D = 10; N = 30; MaxCycle = 1000; runs = 25;
fid = fopen('matlab_results.csv','w');
fprintf(fid,'variant,func,run,best\n');
for v = {'eq15','clip'}
    for k = 1:numel(names)
        if strcmp(v{1},'clip') && strcmp(names{k},'shifted_sphere'), continue; end
        mydata.name = names{k};
        for r = 1:runs
            if strcmp(v{1},'eq15')
                out = algo_MDE_eq15('bench', mydata, N, D, lows(k), ups(k), MaxCycle, r);
            else
                out = algo_MDE_clip('bench', mydata, N, D, lows(k), ups(k), MaxCycle, r);
            end
            fprintf(fid,'%s,%s,%d,%.17g\n', v{1}, names{k}, r, out.bestval);
        end
    end
end
fclose(fid);
