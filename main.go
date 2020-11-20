package main

import (
	"gotf/bert"
	"gotf/http"
)

/* 预训练模型路径 */
const(
	modelPath = "../../../nlp/albert_QA/outputs/saved-model"
	vocabPath = "../../../nlp/nlp_model/albert_zh_base/vocab_chinese.txt"
)

/* 主入口 */
func main() {
	/* 初始化模型 */
	bert.InitModel(modelPath, vocabPath)

	/* 启动server */
	http.RunServer()
}
