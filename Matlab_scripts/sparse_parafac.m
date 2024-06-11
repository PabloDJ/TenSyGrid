%addpath('C:\Users\Usuario\OneDrive\Escritorio\eRoots\tensor_toolbox');
%savepath()
n = 30;
q = 3;
subs = [];
val = [];
dimensions = ones(1,n)*2;
data = load('data/sparse_matrix.mat');
data_coords = data.coords';
data_values = data.values';
m = size(data_coords,2)-1;
p = max(data_coords(end,:))+1;
dimensions_m = [ones(1,m)*2, p];

for i = 1:n
    for j = 1:n
        index = ones(1,n);
        index(i) = 2;
        index(j) = 2;
        subs = [subs; index]; 
        val = [val; rand()]; 
    end
end

rank = 5;

data_coords = data_coords + 1.0;
F = sptensor(subs(1:10,:), val(1:10,:), dimensions);
F = sptensor(data_coords(1,:), data_values(1,:), dimensions_m);
M = cp_arls_lev(F, rank);

folder_path = 'Data';
file_name = 'CP_decomposition.mat';
full_path = fullfile(folder_path, file_name);
save(full_path, 'M');