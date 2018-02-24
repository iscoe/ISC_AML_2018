#-------------------------------------------------------------------------------
# This file serves as an example to show how one might use the example code.
# Of primary interest are:
#
#   baseline-attacks   : Creates example/baseline attacks 
#   baseline-defenses  : Create example/baseline defense submissions
#   demo               : Creates a local competition "simulation" by running
#                        some baseline algorithms against each other.
#
# Feb, 2018
#-------------------------------------------------------------------------------

# Change this to point to your local sample data (with truth)
#GROUND_TRUTH_DIR=~/Data/FMoW_a
GROUND_TRUTH_DIR=/home/fendlnm1/FMOW_1000_CORRECT

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
	cd ./attacks/sample_attack && ./run_attack.sh $(GROUND_TRUTH_DIR) ifgm

baseline-defenses :
	cd ./defenses/sample_defense && ./defense_zip.sh 


# This just runs all the necessary pieces in order
demo : baseline-attacks baseline-defenses dummy-framework eval


dummy-framework :
	echo "creating ground truth"
	mkdir -p $(SIM_DIR)/ground_truth
	cp $(GROUND_TRUTH_DIR)/* $(SIM_DIR)/ground_truth
	@echo "--------------------------"
	@echo "creating noop-defense"
	@echo "--------------------------"
	mkdir -p $(TEAM_DIR)/noop/attack
	mkdir -p $(TEAM_DIR)/noop/defense
	cp ./defenses/sample_defense/defense_noop_submission.zip $(TEAM_DIR)/noop/defense
	@echo "--------------------------"
	@echo "creating averaging defense"
	@echo "--------------------------"
	mkdir -p $(TEAM_DIR)/averaging/attack
	mkdir -p $(TEAM_DIR)/averaging/defense
	cp ./defenses/sample_defense/defense_avg_submission.zip $(TEAM_DIR)/averaging/defense
	@echo "--------------------------"
	@echo "creating random attack"
	@echo "--------------------------"
	mkdir -p $(TEAM_DIR)/random/attack
	mkdir -p $(TEAM_DIR)/random/defense
	cp ./attacks/sample_attack/sample_attack_random.zip $(TEAM_DIR)/random/attack
	@echo "--------------------------"
	@echo "creating FGM attack"
	@echo "--------------------------"
	mkdir -p $(TEAM_DIR)/fgm/attack
	mkdir -p $(TEAM_DIR)/fgm/defense
	cp ./attacks/sample_attack/sample_attack_fgm.zip $(TEAM_DIR)/fgm/attack
	@echo "--------------------------"
	@echo "creating I-FGM attack"
	@echo "--------------------------"
	mkdir -p $(TEAM_DIR)/ifgm/attack
	mkdir -p $(TEAM_DIR)/ifgm/defense	
	cp ./attacks/sample_attack/sample_attack_ifgm.zip $(TEAM_DIR)/ifgm/attack


eval :
	python evaluate_submissions.py $(TEAM_DIR) $(GROUND_TRUTH_DIR) $(OUTPUT_DIR) 1 5 10


# Blows away the example framework
clean :
	\rm -rf $(SIM_DIR)


query :
	python evaluate_submissions.py $(TEAM_DIR) query $(OUTPUT_DIR)



