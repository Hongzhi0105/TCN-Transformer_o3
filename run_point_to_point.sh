
# #########################   42-JAN   #############################

avoid_interval=1   #mounth

Output_folder_path="TCN_ReLU-JAN-42_basecase"
mkdir -p "$Output_folder_path"

site_sets="[42]"

Input_timesteps=72

Output_timesteps=72

python datasetdiff.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets
python train.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval
python inference.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval 

# #########################   42-APR   #############################

avoid_interval=4   #mounth

Output_folder_path="TCN_ReLU-APR-42_basecase"
mkdir -p "$Output_folder_path"

site_sets="[42]"

Input_timesteps=72

Output_timesteps=72

python datasetdiff.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets
python train.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval
python inference.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval 

# #########################   42-JUL   #############################

avoid_interval=7   #mounth

Output_folder_path="TCN_ReLU-JUL-42_basecase"
mkdir -p "$Output_folder_path"

site_sets="[42]"

Input_timesteps=72

Output_timesteps=72

python datasetdiff.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets
python train.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval
python inference.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval 

# #########################   42-OCT   #############################

avoid_interval=10   #mounth

Output_folder_path="TCN_ReLU-OCT-42_basecase"
mkdir -p "$Output_folder_path"

site_sets="[42]"

Input_timesteps=72

Output_timesteps=72

python datasetdiff.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets
python train.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval
python inference.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval # #########################   43-JAN   #############################


# #########################   43-JAN   #############################

avoid_interval=1   #mounth

Output_folder_path="TCN_ReLU-JAN-43_basecase"
mkdir -p "$Output_folder_path"

site_sets="[43]"

Input_timesteps=72

Output_timesteps=72

python datasetdiff.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets
python train.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval
python inference.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval 

# #########################   43-APR   #############################

avoid_interval=4   #mounth

Output_folder_path="TCN_ReLU-APR-43_basecase"
mkdir -p "$Output_folder_path"

site_sets="[43]"

Input_timesteps=72

Output_timesteps=72

python datasetdiff.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets
python train.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval
python inference.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval 

# #########################   43-JUL   #############################

avoid_interval=7   #mounth

Output_folder_path="TCN_ReLU-JUL-43_basecase"
mkdir -p "$Output_folder_path"

site_sets="[43]"

Input_timesteps=72

Output_timesteps=72

python datasetdiff.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets
python train.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval
python inference.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval 

# #########################   43-OCT   #############################

avoid_interval=10   #mounth

Output_folder_path="TCN_ReLU-OCT-43_basecase"
mkdir -p "$Output_folder_path"

site_sets="[43]"

Input_timesteps=72

Output_timesteps=72

python datasetdiff.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets
python train.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval
python inference.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval # #########################   43-JAN   #############################


# #########################   44-JAN   #############################

avoid_interval=1   #mounth

Output_folder_path="TCN_ReLU-JAN-44_basecase"
mkdir -p "$Output_folder_path"

site_sets="[44]"

Input_timesteps=72

Output_timesteps=72

python datasetdiff.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets
python train.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval
python inference.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval 

# #########################   44-APR   #############################

avoid_interval=4   #mounth

Output_folder_path="TCN_ReLU-APR-44_basecase"
mkdir -p "$Output_folder_path"

site_sets="[44]"

Input_timesteps=72

Output_timesteps=72

python datasetdiff.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets
python train.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval
python inference.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval 

# #########################   44-JUL   #############################

avoid_interval=7   #mounth

Output_folder_path="TCN_ReLU-JUL-44_basecase"
mkdir -p "$Output_folder_path"

site_sets="[44]"

Input_timesteps=72

Output_timesteps=72

python datasetdiff.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets
python train.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval
python inference.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval 

# #########################   44-OCT   #############################

avoid_interval=10   #mounth

Output_folder_path="TCN_ReLU-OCT-44_basecase"
mkdir -p "$Output_folder_path"

site_sets="[44]"

Input_timesteps=72

Output_timesteps=72

python datasetdiff.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets
python train.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval
python inference.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval # #########################   43-JAN   #############################


# #########################   45-JAN   #############################

avoid_interval=1   #mounth

Output_folder_path="TCN_ReLU-JAN-45_basecase"
mkdir -p "$Output_folder_path"

site_sets="[45]"

Input_timesteps=72

Output_timesteps=72

python datasetdiff.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets
python train.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval
python inference.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval 

# #########################   45-APR   #############################

avoid_interval=4   #mounth

Output_folder_path="TCN_ReLU-APR-45_basecase"
mkdir -p "$Output_folder_path"

site_sets="[45]"

Input_timesteps=72

Output_timesteps=72

python datasetdiff.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets
python train.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval
python inference.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval 

# #########################   45-JUL   #############################

avoid_interval=7   #mounth

Output_folder_path="TCN_ReLU-JUL-45_basecase"
mkdir -p "$Output_folder_path"

site_sets="[45]"

Input_timesteps=72

Output_timesteps=72

python datasetdiff.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets
python train.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval
python inference.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval 

# #########################   45-OCT   #############################

avoid_interval=10   #mounth

Output_folder_path="TCN_ReLU-OCT-45_basecase"
mkdir -p "$Output_folder_path"

site_sets="[45]"

Input_timesteps=72

Output_timesteps=72

python datasetdiff.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets
python train.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval
python inference.py --Input_timestep $Input_timesteps --Output_timestep $Output_timesteps --output_dir $Output_folder_path --site_sets $site_sets --Avoid_interval $avoid_interval # #########################   43-JAN   #############################
