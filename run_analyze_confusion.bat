@echo off
echo 分析模型混淆矩阵，特别关注5和6的混淆情况...
cd src
python analyze_confusion.py
pause

