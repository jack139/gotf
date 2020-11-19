package main

import (
	"fmt"

	"github.com/buckhx/gobert/tokenize"
	"github.com/buckhx/gobert/tokenize/vocab"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/aclements/go-gg/generic/slice"
)

func main() {
	corpus := "深度学习的基础是机器学习中的分散表示（distributed representation）。分散表示假定观测值是由不同因" +
		"子相互作用生成。在此基础上，深度学习进一步假定这一相互作用的过程可分为多个层次，代表对观测值的多层抽象。不同" +
		"的层数和层的规模可用于不同程度的抽象。深度学习运用了这分层次抽象的思想，更高层次的概念从低层次的概念学习得到。" +
		"这一分层结构常常使用贪心算法逐层构建而成，并从中选取有助于机器学习的更有效的特征。不少深度学习算法都以无监督" +
		"学习的形式出现，因而这些算法能被应用于其他算法无法企及的无标签数据，这一类数据比有标签数据更丰富，也更容易获" +
		"得。这一点也为深度学习赢得了重要的优势。"

	question := "深度学习的优点是什么？"

	modelPath := "../../../nlp/albert_QA/outputs/saved-model"
	vocabPath := "../../../nlp/nlp_model/albert_zh_base/vocab_chinese.txt"
	voc, err := vocab.FromFile(vocabPath)
	if err != nil {
		panic(err)
	}
	m, err := tf.LoadSavedModel(modelPath, []string{"train"}, nil)
	if err != nil {
		panic(err)
	}

	tkz := tokenize.NewTokenizer(voc)
	ff := tokenize.FeatureFactory{Tokenizer: tkz, SeqLen: 512}
	// 拼接输入
	input_tokens := question + tokenize.SequenceSeparator + corpus
	// 获取 token 向量
	f := ff.Feature(input_tokens)

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

	fmt.Println(f.Tokens)
	fmt.Println(f.TokenIDs)
	fmt.Println(f.Mask)
	fmt.Println(f.TypeIDs)

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
	//fmt.Println("Value", res[0].Value().([][]float32))

	fmt.Println("DataType", res[1].DataType())
	fmt.Println("Shape", res[1].Shape())
	//fmt.Println("Value", res[1].Value().([][]float32))

	st := slice.ArgMax(res[0].Value().([][]float32)[0])
	ed := slice.ArgMax(res[1].Value().([][]float32)[0])
	fmt.Println(st, ed)
	fmt.Println(input_tokens[st:ed+1])

}
