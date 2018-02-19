#-------------------------------------------------------------------------------
# This file serves as an example to show how one might use the example code.
# Of primary interest are:
#
#   baseline-attack   : Demonstrates how to generate AE using CleverHans (CH)
#   baseline-defense  : Demonstrates how one might attempt to classify images.
#   eval              : Demonstrates evaluation framework.
#
# Feb, 2018
#-------------------------------------------------------------------------------

# Directories needed for evaluation
SUBMISSION_DIR=../misc/simulated_environment/submissions
GROUND_TRUTH_DIR=../misc/simulated_environment/truth
OUTPUT_DIR=./Evaluation_Output

#-------------------------------------------------------------------------------

default :
	echo "Please pick an explicit target!"


# Note: the mkdir calls here are to help out evaluation; if you are a 
#       performer/participant in competition, you can ignore this.
baseline-attack:
	mkdir -p $(SUBMISSION_DIR)/team_a/attack
	mkdir -p $(SUBMISSION_DIR)/team_a/defense
	echo "TODO: run attack on 1000 images!"


baseline-defense :
	mkdir -p $(SUBMISSION_DIR)/team_b/attack
	mkdir -p $(SUBMISSION_DIR)/team_b/defense
	cd ./defenses/sample_defense_submission/ && ./defense_zip.sh 
	mv ./defenses/sample_defense_submission/defense_noop_submission.zip $(SUBMISSION_DIR)/team_b/defense


eval :
	python evaluate_submissions.py $(SUBMISSION_DIR) $(GROUND_TRUTH_DIR) $(OUTPUT_DIR) 10 20 50
