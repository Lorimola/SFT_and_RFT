export CUDA_VISIBLE_DEVICES=0

echo "Start Running"


echo "Running coreset.py"
python3 /data2/home/donglingzhong/yangsb/SAR/Code/coreset.py
echo "coreset.py finish"


echo "Running script.py"
python3 /data2/home/donglingzhong/yangsb/SAR/Code/script.py --input_file=/data2/home/donglingzhong/yangsb/Dateset/Original/AITZ/train.json --output_dir=/data2/home/donglingzhong/yangsb/Dateset/Train/AITZ --dataset_type="aitz"

python3 /data2/home/donglingzhong/yangsb/SAR/Code/script.py --input_file=/data2/home/donglingzhong/yangsb/Dateset/Subset/train.json --output_dir=/data2/home/donglingzhong/yangsb/Dateset/Train/android_control --dataset_type="control"
echo "script.py finish"


echo "Running train_sft.py"
python3 /data2/home/donglingzhong/yangsb/SAR/Code/train_sft.py --model_name_or_path=/data4/models_wzr/Qwen2.5-VL-7B-Instruct --dataset_path=/data2/home/donglingzhong/yangsb/Dateset/Train/AITZ --output_dir=/data2/home/donglingzhong/yangsb/Models/SFT/AITZ --num_train_epochs=3 --learning_rate=2e-5 --bf16=True --use_lora=False

python3 /data2/home/donglingzhong/yangsb/SAR/Code/train_sft.py --model_name_or_path=/data4/models_wzr/Qwen2.5-VL-7B-Instruct --dataset_path=/data2/home/donglingzhong/yangsb/Dateset/Train/android_control --output_dir=/data2/home/donglingzhong/yangsb/Models/SFT/android_control --num_train_epochs=3 --learning_rate=2e-5 --bf16=True --use_lora=False
echo "train_sft.py finish"


echo "Running train_rft.py"
python3 /data2/home/donglingzhong/yangsb/SAR/Code/train_rft.py --model_name_or_path=/data4/models_wzr/Qwen2.5-VL-7B-Instruct --dataset_path=/data2/home/donglingzhong/yangsb/Dateset/Train/AITZ --output_dir=/data2/home/donglingzhong/yangsb/Models/RFT/AITZ --learning_rate=1e-6 --num_train_epochs=3 --bf16=True --use_qlora=False

python3 /data2/home/donglingzhong/yangsb/SAR/Code/train_rft.py --model_name_or_path=/data4/models_wzr/Qwen2.5-VL-7B-Instruct --dataset_path=/data2/home/donglingzhong/yangsb/Dateset/Train/android_control --output_dir=/data2/home/donglingzhong/yangsb/Models/RFT/android_control --learning_rate=1e-6 --num_train_epochs=3 --bf16=True --use_qlora=False
echo "train_rft.py finish"


echo "Running eval.py"
python3 /data2/home/donglingzhong/yangsb/SAR/Code/eval.py --model_path=/data2/home/donglingzhong/yangsb/Models/SFT/AITZ --output_dir=/data2/home/donglingzhong/yangsb/Eval/SFT/AITZ --data_name='aitz_test'

python3 /data2/home/donglingzhong/yangsb/SAR/Code/eval.py --model_path=/data2/home/donglingzhong/yangsb/Models/SFT/AITZ --output_dir=/data2/home/donglingzhong/yangsb/Eval/SFT/AITZ --data_name='android_control_high_test'

python3 /data2/home/donglingzhong/yangsb/SAR/Code/eval.py --model_path=/data2/home/donglingzhong/yangsb/Models/RFT/AITZ --output_dir=/data2/home/donglingzhong/yangsb/Eval/RFT/AITZ --data_name='aitz_test'

python3 /data2/home/donglingzhong/yangsb/SAR/Code/eval.py --model_path=/data2/home/donglingzhong/yangsb/Models/RFT/AITZ --output_dir=/data2/home/donglingzhong/yangsb/Eval/RFT/AITZ --data_name='android_control_high_test'

python3 /data2/home/donglingzhong/yangsb/SAR/Code/eval.py --model_path=/data2/home/donglingzhong/yangsb/Models/SFT/android_control --output_dir=/data2/home/donglingzhong/yangsb/Eval/SFT/android_control --data_name='aitz_test'

python3 /data2/home/donglingzhong/yangsb/SAR/Code/eval.py --model_path=/data2/home/donglingzhong/yangsb/Models/SFT/android_control --output_dir=/data2/home/donglingzhong/yangsb/Eval/SFT/android_control --data_name='android_control_high_test'

python3 /data2/home/donglingzhong/yangsb/SAR/Code/eval.py --model_path=/data2/home/donglingzhong/yangsb/Models/RFT/android_control --output_dir=/data2/home/donglingzhong/yangsb/Eval/RFT/android_control --data_name='aitz_test'

python3 /data2/home/donglingzhong/yangsb/SAR/Code/eval.py --model_path=/data2/home/donglingzhong/yangsb/Models/RFT/android_control --output_dir=/data2/home/donglingzhong/yangsb/Eval/RFT/android_control --data_name='android_control_high_test'
echo "Eval Finishes!"