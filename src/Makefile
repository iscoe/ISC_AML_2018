#-------------------------------------------------------------------------------
# This file serves as an example to show how one might use the example code.
# Of primary interest are:
#
#   baseline-attack   : Demonstrates how to generate AE using CleverHans (CH)
#   baseline-defense  : Demonstrates how one might attempt to classify images.
#
#   dummy-framework   : Creates a local simulation of inputs from
#                       a few contenstants (for debugging)
#                       Requires one run the baseline attacks/defenses first.
#
#   eval              : Demonstrates evaluation framework.
#                       Requires running dummy-framework first.
#
# Feb, 2018
#-------------------------------------------------------------------------------

SIM_DIR=./simulated_environment
TEAM_DIR=$(SIM_DIR)/submissions
GROUND_TRUTH_DIR=/home/fendlnm1/FMOW_1000_CORRECT
ATTACK_ZIP_DIR=attacks/sample_attack/submission_run
#GROUND_TRUTH_DIR=/home/fendlnm1/Fendley/adversarial/ISC_AML_2018/image_sets/FMOW_1000_CORRECT_2

OUTPUT_DIR=./Evaluation_Output

#-------------------------------------------------------------------------------

default :
	echo "Please pick an explicit target!"

iter-attack :
	cd ./attacks/IFGM_attack && ./run_attack.sh $(GROUND_TRUTH_DIR)

baseline-attack :
	cd ./attacks/sample_attack && ./run_attack.sh $(GROUND_TRUTH_DIR)


baseline-defense :
	cd ./defenses/sample_defense_submission/ && ./defense_zip.sh 


dummy-framework :
	echo "creating ground truth"
	mkdir -p $(SIM_DIR)/ground_truth
	cp $(GROUND_TRUTH_DIR)/* $(SIM_DIR)/ground_truth
	echo "creating team A"
	mkdir -p $(TEAM_DIR)/team_a/attack
	mkdir -p $(TEAM_DIR)/team_a/defense
	cp ./defenses/sample_defense_submission/defense_noop_submission.zip $(TEAM_DIR)/team_a/defense
	cp ./attacks/sample_attack/sample_attack_fgm.zip $(TEAM_DIR)/team_a/attack
	echo "creating team B"
	mkdir -p $(TEAM_DIR)/team_b/attack
	mkdir -p $(TEAM_DIR)/team_b/defense	
	cp ./defenses/sample_defense_submission/defense_noop_submission.zip $(TEAM_DIR)/team_b/defense
	echo "creating team C"
	mkdir -p $(TEAM_DIR)/team_c/attack
	mkdir -p $(TEAM_DIR)/team_c/defense
	cp ./defenses/sample_defense_submission/defense_noop_submission.zip $(TEAM_DIR)/team_c/defense

demo : baseline-attack iter-attack baseline-defense dummy-framework eval

eval :
	python evaluate_submissions.py $(TEAM_DIR) $(GROUND_TRUTH_DIR) $(OUTPUT_DIR) 1 5

query :
	python evaluate_submissions.py $(TEAM_DIR) query $(OUTPUT_DIR)



