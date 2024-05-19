clc;
clear;
close all;

addpath('..\..\OpticalFlow\toolbox\epc');

inter_mode = 0;

%%
%%%%%%%%%%%%%%%mse+RGB%%%%%%%%%%%%%%%%%%%%%
%
result_path= '.\results\feature';
classification_results = load(fullfile(result_path, 'devel_fc_label_basic_timesformer_mse_multiscale_cnn_rgb_mse_epoch_1.mat'));
dec_devel_mse = classification_results.outputs_texture_pulse_mse;
dec_devel_similarity = classification_results.outputs_texture_pulse_similarity;
% fc = dec_devel;
% fc_exp = exp(fc);
% fc_exp_sum = repmat(sum(fc_exp, 2), 1, 2);
% fc_softmax = fc_exp ./ fc_exp_sum;
% dec_devel = fc_softmax;
% dec_devel = dec_devel(:, 1);
label_devel = classification_results.label;
classification_results = load(fullfile(result_path, 'test_fc_label_basic_timesformer_mse_multiscale_cnn_rgb_mse_epoch_0.mat'));
dec_test_mse = classification_results.outputs_texture_pulse_mse;
dec_test_similarity = classification_results.outputs_texture_pulse_similarity;
% fc = dec_test;
% fc_exp = exp(fc);
% fc_exp_sum = repmat(sum(fc_exp, 2), 1, 2);
% fc_softmax = fc_exp ./ fc_exp_sum;
% dec_test = fc_softmax;
% dec_test = dec_test(:, 1);
label_test = classification_results.label;

% fuse pixel and similarity
dec_devel_similarity_mse = dec_devel_mse;% .* (1 - dec_devel_similarity);
dec_test_similarity_mse = dec_test_mse;% .* (1 - dec_test_similarity);
[com1.epc.dev, com1.epc.eva, epc_cost] = epc(dec_devel_similarity_mse(label_devel == 0), dec_devel_similarity_mse(label_devel == 1), dec_test_similarity_mse(label_test == 0), dec_test_similarity_mse(label_test == 1), 1, [0.5 0.5]);
HTER_similarity_mse = com1.epc.eva.hter_apri(1);
[com2.epc.dev, com2.epc.eva, epc_cost] = epc(dec_devel_similarity_mse(label_devel == 0), dec_devel_similarity_mse(label_devel == 1), dec_devel_similarity_mse(label_devel == 0), dec_devel_similarity_mse(label_devel == 1), 1, [0.5 0.5]);
EER_similarity_mse = com2.epc.dev.wer_apost(1);
[com2.epc.dev, com2.epc.eva, epc_cost] = epc(dec_test_similarity_mse(label_test == 0), dec_test_similarity_mse(label_test == 1), dec_test_similarity_mse(label_test == 0), dec_test_similarity_mse(label_test == 1), 1, [0.5 0.5]);
EER_similarity_mse = com2.epc.dev.wer_apost(1);
threshold = com2.epc.dev.thrd_fv;
% APCER
dec_attack_test = dec_test_similarity_mse(label_test == 1);
error_attack_classify = dec_attack_test < threshold;
APCER_similarity_mse = sum(error_attack_classify(:)) / length(dec_attack_test);
% BPCER
dec_real_test = dec_test_similarity_mse(label_test == 0);
error_real_classify = dec_real_test > threshold;
BPCER_similarity_mse = sum(error_real_classify(:)) / length(dec_real_test);
% ACER
ACER_similarity_mse = (APCER_similarity_mse + BPCER_similarity_mse) / 2;
