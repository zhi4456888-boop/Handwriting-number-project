@echo off
echo 训练专门针对5和6混淆问题的模型...
echo 使用注意力机制模型（推荐）
cd src
python train_focused_5_6.py attention
pause

