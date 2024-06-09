close all;
clear;
clc;
% This script measures the time between each configuration of tx and rx
% saves the enlapsed time for each config into .csv format

%-------------------------------------------------------------------------%
% Site Definition & initialization parameters                            
%-------------------------------------------------------------------------%

% Filepaths of the outputdata and filepath of the model
result_folderpath = "carla_multiple_rx/";
model = "buildings_hd_mesh_simplified_with_terrain.stl";


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

        [viewer, tx_cellarray, rx_cellarray] = init_site("simulation_settings/tx_positions.csv", ...
        "simulation_settings/rx_positions.csv", model);
        % disp("site loaded");

        rx_array = [rx_cellarray{1} rx_cellarray{2}];
        for index = 3:51
            rx_array = [rx_array rx_cellarray{index}];
        end

        rtpm.MaxNumReflections = k ;
    
        M = zeros(length(tx_cellarray), 1); % array used to store execution time
        for i=1:length(tx_cellarray) % for each transmitter
            d1 = datetime;
            ray = raytrace(tx_cellarray{i}, rx_array, rtpm);
            M(i) = seconds(datetime - d1); % store enlapsed time in seconds
        end
        out_str = strcat("writing data for r: ", string(k), ", angular separation: ", angular_separation(l));
        disp(out_str)
        
        % write the data
        file_name = strcat(folder_name, 'MaxNumReflections_', int2str(rtpm.MaxNumReflections), '.csv');
        writematrix(M, file_name)  

        viewer.close()
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
    % show(tx_array{i}); % render the added site in site-viewer
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
    % show(rx_array{i}); % render the added site in site-viewer
end
end