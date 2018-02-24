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

# this should point to whatever sample data you are using
GROUND_TRUTH_DIR=~/Data/FMoW_a
#GROUND_TRUTH_DIR=/home/fendlnm1/FMOW_1000_CORRECT

# these are temporary directories for demo
SIM_DIR=./simulated_environment
TEAM_DIR=$(SIM_DIR)/submissions

OUTPUT_DIR=./Evaluation_Output

#-------------------------------------------------------------------------------

default :
	echo "Please pick an explicit target!"


baseline-attacks :
	cd ./attacks/sample_attack && ./run_attack.sh $(GROUND_TRUTH_DIR) fgm
	cd ./attacks/sample_attack && ./run_attack.sh $(GROUND_TRUTH_DIR) random

baseline-defenses :
	cd ./defenses/sample_defense && ./defense_zip.sh 


dummy-framework :
	echo "creating ground truth"
	mkdir -p $(SIM_DIR)/ground_truth
	cp $(GROUND_TRUTH_DIR)/* $(SIM_DIR)/ground_truth
	@echo "--------------------------"
	@echo "creating baseline_1"
	@echo "--------------------------"
	mkdir -p $(TEAM_DIR)/baseline_1/attack
	mkdir -p $(TEAM_DIR)/baseline_1/defense
	cp ./defenses/sample_defense/defense_noop_submission.zip $(TEAM_DIR)/baseline_1/defense
	cp ./attacks/sample_attack/sample_attack_fgm.zip $(TEAM_DIR)/baseline_1/attack
	@echo "--------------------------"
	@echo "creating baseline_2"
	@echo "--------------------------"
	mkdir -p $(TEAM_DIR)/baseline_2/attack
	mkdir -p $(TEAM_DIR)/baseline_2/defense	
	cp ./defenses/sample_defense/defense_avg_submission.zip $(TEAM_DIR)/baseline_2/defense
	cp ./attacks/sample_attack/sample_attack_random.zip $(TEAM_DIR)/baseline_2/attack

eval :
	python evaluate_submissions.py $(TEAM_DIR) $(GROUND_TRUTH_DIR) $(OUTPUT_DIR) 1 3

# This just runs all the necessary pieces in order
demo : baseline-attacks baseline-defenses dummy-framework eval

# Blows away the example framework
clean :
	\rm -rf $(SIM_DIR)


query :
	python evaluate_submissions.py $(TEAM_DIR) query $(OUTPUT_DIR)



