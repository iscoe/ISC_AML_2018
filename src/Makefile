

SUBMISSION_DIR=../misc/simulated_environment/submissions
TRUTH_DIR=../misc/simulated_environment/truth



default :
	echo "Please pick an explicit target!"


# Construct a simulated set of submissions.
# This is just for testing/demonstration purposes.
# 
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
	python evaluate_submissions.py $(SUBMISSION_DIR) $(TRUTH_DIR) .005 .01
