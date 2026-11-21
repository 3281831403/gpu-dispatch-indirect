# gpu-dispatch-indirect

GPU 间接绘制参数打包与批次调度的修复题。仓库开局是纯桩环境：`indirect/args.py` 里的 8 个函数全部 `raise NotImplementedError`，15 个测试全部红，且必须干净收集。

## ① 题面

改成间接绘制后，批次参数错位、偶尔还画出满屏三角形：打包用大端导致 GPU 读到的整数全错；结构体 20 字节却按 16 对齐导致后续命令整体错位；stride 只取 20 的倍数却不校验最小值导致越界写；偏移加到 count 而不是 first/base 导致绘制范围错误；分块只按命令数而不看字节预算导致单块超限；count 为 0 不校验产生空绘制；负偏移不报错导致索引回绕；解包不校验长度读到下一个命令；批量共用参数缓冲导致批次之间串数据；alignment=0 不校验导致除零。每条命令使用 {"count","instance","first","base_vertex"}，参数缓冲按 4 字节对齐、每条命令固定 20 字节，容差 1 字节。indirect/args.py 是核心，tests/test_args.py 15个红。要修的是：小端 20 字节打包、stride 最小值与对齐、first/base 偏移、按字节预算分块、count/负偏移/长度校验、每批独立参数缓冲和 alignment 校验，容差 1 字节。

## ② 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始状态的输出尾部必须恰好是：

```text
15 failed in <时间>s
```

即恰好 15 个 `failed`，且没有 collection error、import error 或 syntax error。环境仅使用标准库（`struct`、`math`）和 pytest，离线运行，不依赖随机数。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## ④ tests 不可修改

`tests/test_args.py` 不可修改：不得删改测试、增加 skip/xfail、放宽断言，也不得通过测试侧的特殊分支绕过断言。修复应全部落在 `indirect/args.py`，并保持 `IndirectError` 的异常契约，以及 4 字节对齐、20 字节命令、stride、偏移和字节预算的 docstring 约定。
