package http

import (
	"gotf/bert"

	"log"
	"time"
	"encoding/json"

	"github.com/fasthttp/router"
	"github.com/valyala/fasthttp"
	"github.com/AubSs/fasthttplogger"
)

/* 返回值的 content-type */
var (
	strContentType = []byte("Content-Type")
	strApplicationJSON = []byte("application/json")
)

/* 处理返回值，返回json */
func doJSONWrite(ctx *fasthttp.RequestCtx, code int, obj interface{}) {
	ctx.Response.Header.SetCanonical(strContentType, strApplicationJSON)
	ctx.Response.SetStatusCode(code)
	start := time.Now()
	if err := json.NewEncoder(ctx).Encode(obj); err != nil {
		elapsed := time.Since(start)
		log.Printf("", elapsed, err.Error(), obj)
		ctx.Error(err.Error(), fasthttp.StatusInternalServerError)
	}
}

/* 
	使用 tf 进行问答，输入格式： 
		{"c":"背景知识", "q":"问题"}
*/
func onQA(ctx *fasthttp.RequestCtx) {
	retJson := map[string] string {"code":"9000","msg":"error"}
	content := ctx.PostBody()
	fields := make(map[string]interface{})
	if err := json.Unmarshal(content, &fields); err != nil {
		retJson["code"] = "9001"; retJson["msg"] = "invalid json data"
		doJSONWrite(ctx, fasthttp.StatusOK, retJson)
		return
	}
	log.Printf("%v", fields)

	/* 检查 c q 是否存在 */
	corpus, ok := fields["c"]
	if !ok {
		retJson["code"] = "9010"; retJson["msg"] = "data error"
		doJSONWrite(ctx, fasthttp.StatusOK, retJson)
		return
	}
	question, ok := fields["q"]
	if !ok {
		retJson["code"] = "9020"; retJson["msg"] = "data error"
		doJSONWrite(ctx, fasthttp.StatusOK, retJson)
		return
	}

	/* 检查类型是否是字符串 */
	if _, ok := corpus.(string); !ok {
		retJson["code"] = "9030"; retJson["msg"] = "data type error"
		doJSONWrite(ctx, fasthttp.StatusOK, retJson)
		return		
	}
	if _, ok := question.(string); !ok {
		/* not string */
		retJson["code"] = "9040"; retJson["msg"] = "data type error"
		doJSONWrite(ctx, fasthttp.StatusOK, retJson)
		return
	}

	/* 调用问答 */
	ans := bert.BertQA(corpus.(string), question.(string))
	log.Printf("%v", ans)
	doJSONWrite(ctx, fasthttp.StatusOK, map[string] string {"code":"0","msg":"ok","data":ans})
}

/* 测试 */
func returnJSON(ctx *fasthttp.RequestCtx) {
	countryCapitalMap := map[string] string {"France":"Paris","Italy":"Rome"}
	doJSONWrite(ctx, fasthttp.StatusOK, countryCapitalMap)
}

/* 根返回 */
func index(ctx *fasthttp.RequestCtx) {
	log.Printf("%v", ctx.RemoteAddr())
	ctx.WriteString("Hello world.")
}


/* 入口 */
func RunServer() {
	/* router */
	r := router.New()
	r.GET("/", index)
	r.GET("/test", returnJSON)
	r.POST("/qa", onQA)

	/* 启动server */
	s := &fasthttp.Server{
		Handler: fasthttplogger.Combined(r.Handler),
		Name: "FastHttpLogger",
	}
	log.Fatal(s.ListenAndServe(":8080"))
}
