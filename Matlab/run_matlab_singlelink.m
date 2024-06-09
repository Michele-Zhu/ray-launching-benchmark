close all;
clear;
clc;
% This script measures the time between each configuration of tx and rx
% saves the enlapsed time for each config into .csv format

%-------------------------------------------------------------------------%
% Site Definition & initialization parameters                            
%-------------------------------------------------------------------------%

% Filepaths of the outputdata and filepath of the model
result_folderpath = "carla_map_simulation_time_data/";
model = "buildings_hd_mesh_simplified_with_terrain.stl";
[viewer, tx_array, rx_array] = init_site("simulation_settings/tx_positions.csv", ...
    "simulation_settings/rx_positions.csv", model);

%-------------------------------------------------------------------------%
% Propagation Model
%-------------------------------------------------------------------------%

% change method and number of reflections and time the raytrace method
% ray tracing propagation model using SBR method
rtpm = propagationModel("raytracing", ...
    Method = "sbr", ...
    AngularSeparation = "high", ... % angular separation between launched rays using sbr
    MaxNumReflections = 1, ... % number of path reflections
    MaxNumDiffractions = 1, ...
    CoordinateSystem = "cartesian", ...
    SurfaceMaterial = "concrete", ... 
    TerrainMaterial = "concrete");

%-------------------------------------------------------------------------%
% Ray Trace
%-------------------------------------------------------------------------%

angular_separation = ["high", "medium", "low"];
for l = 1:length(angular_separation) % iterate over different angular separation settings
    rtpm.AngularSeparation = angular_separation(l);
    folder_name = strcat(result_folderpath, 'angular_separation_', rtpm.AngularSeparation, '/');
    mkdir(folder_name);
    
    for k = 1:10 % loop over different number of max reflections
        rtpm.MaxNumReflections = k ;
    
        M = zeros(length(rx_array), length(tx_array)); % matrix used to store execution time
        for i=1:length(tx_array) % for each transmitter
            for j=1:length(rx_array) % for each receiver
                d1 = datetime;
                ray = raytrace(tx_array{i}, rx_array{j}, rtpm);
                M(j, i) = seconds(datetime - d1); % store enlapsed time in seconds
            end
        end
        out_str = strcat("writing data for r: ", string(k), ", angular separation: ", angular_separation(l));
        disp(out_str)
        
        % save M in a table and write it in csv
        T = array2table(M);
        
        % Write row names
        rx_names = cell(length(rx_array), 1);
        for i=1:length(rx_array)
            rx_names{i} = rx_array{i}.Name;
        end
        T.Properties.RowNames = rx_names;
        % Write col names
        tx_names =  cell(length(tx_array), 1);
        for i = 1:length(tx_array)
            tx_names{i} = tx_array{i}.Name;
        end
        T.Properties.VariableNames = tx_names;
        
        % write the data
        file_name = strcat(folder_name, 'MaxNumReflections_', int2str(rtpm.MaxNumReflections), '.csv');
        % writetable(T, file_name, 'writeRowNames', true)  
    end
end

%% Utility functions
function [viewer, tx_array, rx_array] = init_site(tx_position_filename, rx_position_filename, model_filename)
viewer = siteviewer(SceneModel=model_filename,...
    ShowOrigin = 0);

tx_M = readmatrix(tx_position_filename);
tx_array = cell(length(tx_M), 1);

% create tx sites
for i=1:length(tx_M)
    tx_name = strcat("tx_", int2str(i));
    tx_position = tx_M(i, :).'; % x' is the complex conjugate transpose (adjoint matrix), x.' is the transpose
    tx_array{i} = txsite(Name = tx_name, ...
        CoordinateSystem = "cartesian", ...
        AntennaPosition = tx_position, ...
        TransmitterPower = 1, ...
        TransmitterFrequency= 28e9);
    show(tx_array{i}); % render the added site in site-viewer
end

rx_M = readmatrix(rx_position_filename);
rx_array = cell(length(rx_M), 1);

% create rx sites
for i=1:length(rx_M)
    rx_name = strcat("rx_", int2str(i));
    rx_position = rx_M(i, :).';
    rx_array{i} = rxsite(Name = rx_name, ...
        CoordinateSystem = "cartesian", ...
        Antenna = "isotropic", ...
        AntennaPosition = rx_position);
    show(rx_array{i}); % render the added site in site-viewer
end
end