# text_classifier_bert

基于 Bert 的文本分类实践

使用 [Pytorch](https://pytorch.org/get-started/locally/#start-locally) + [Transformers](https://github.com/huggingface/transformers) 框架


进入 Bert 目录，执行 如下类似命令即可：

```
python run.py \
--model='bert' \
--data_dir='dir' \
--model_name_or_path='dir' \
--output_dir='./output' \
--config_name='dir/config.json' \

```

具体参数可以参照 `run.py` 文件内的 args 设置函数。
