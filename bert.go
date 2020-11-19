package main

import (
	"fmt"

	"github.com/buckhx/gobert/tokenize"
	"github.com/buckhx/gobert/tokenize/vocab"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	modelPath := "../../../nlp/albert_QA/outputs/saved-model"
	vocabPath := "../../../nlp/nlp_model/albert_zh_base/vocab_chinese.txt"
	voc, err := vocab.FromFile(vocabPath)
	if err != nil {
		panic(err)
	}
	tkz := tokenize.NewTokenizer(voc)
	ff := tokenize.FeatureFactory{Tokenizer: tkz, SeqLen: 512}
	f := ff.Feature("这是什么")
	m, err := tf.LoadSavedModel(modelPath, []string{"train"}, nil)
	if err != nil {
		panic(err)
	}
	tids, err := tf.NewTensor([][]int32{f.TokenIDs})
	if err != nil {
		panic(err)
	}
	new_mask := make([]float32, len(f.Mask))
	for i, v := range f.Mask {
		new_mask[i] = float32(v)
	}
	mask, err := tf.NewTensor([][]float32{new_mask})
	if err != nil {
		panic(err)
	}
	sids, err := tf.NewTensor([][]int32{f.TypeIDs})
	if err != nil {
		panic(err)
	}

	fmt.Println([][]int32{f.TokenIDs})
	fmt.Println([][]float32{new_mask})
	fmt.Println([][]int32{f.TypeIDs})

	res, err := m.Session.Run(
		map[tf.Output]*tf.Tensor{
			m.Graph.Operation("input_ids").Output(0):      tids,
			m.Graph.Operation("input_mask").Output(0):     mask,
			m.Graph.Operation("segment_ids").Output(0):    sids,
		},
		[]tf.Output{
			m.Graph.Operation("finetune_mrc/Squeeze").Output(0),
			m.Graph.Operation("finetune_mrc/Squeeze_1").Output(0),
		},
		nil,
	)
	if err != nil {
		panic(err)
	}
	fmt.Println("DataType", res[0].DataType())
	fmt.Println("Shape", res[0].Shape())
	fmt.Println("Value", res[0].Value().([][]float32))
}
