### AutoGen实战 
本节课，我们深入探索autogen的各种能力，包括：

一、检索增强的聊天

&emsp;&emsp;1.1 使用RetrieveChat进行检索增强代码生成和问答

&emsp;&emsp;1.2 RetrieveChat实例

&emsp;&emsp;&emsp;&emsp;应用1：无需人类反馈的检索聊天：基于文档生成代码

&emsp;&emsp;&emsp;&emsp;应用2：无需人类反馈的检索聊天：基于文档回答问题

&emsp;&emsp;&emsp;&emsp;应用3：含人类反馈的检索聊天：基于文档生成代码

&emsp;&emsp;&emsp;&emsp;应用4：含人类反馈的检索聊天：基于文档回答问题

&emsp;&emsp;&emsp;&emsp;应用5：含人类反馈的检索聊天：基于文档回答问题

&emsp;&emsp;1.3 定制RetrieveUserProxyAgent 

&emsp;&emsp;&emsp;&emsp;1.3.1 定制嵌入函数 

&emsp;&emsp;&emsp;&emsp;1.3.2 定制文本分割函数 

&emsp;&emsp;&emsp;&emsp;1.3.3 定制向量数据库  

&emsp;&emsp;&emsp;&emsp;1.4 RetrieveChat实战：自动群聊

二、教会大模型新的技能：TeachableAgent

&emsp;&emsp;应用1：学习用户信息

&emsp;&emsp;应用2：学习新的事实

&emsp;&emsp;应用3：学习用户偏好

&emsp;&emsp;应用4：学习新的技能

三、自动构建的多智能体系统

四、使用代码解释器的GPTAssistant

&emsp;&emsp;应用1：数学问题求解

&emsp;&emsp;应用2：使用代码解释器绘图

五、再探群聊

&emsp;&emsp;5.1 使用select_speaker实现多层次的信息流

&emsp;&emsp;5.2 使用图结构建模聊天路径


&emsp;&emsp;本节课我们介绍AutoGen在实际使用中比较有用的功能：利用RAG技术来增强LLM使用外部知识的能力，从而拓宽LLM的能力范围。介绍TeachableAgent，来持久化LLM学习到的知识，这样能减少一些重复的输入提示词的工作。随后介绍AutoGen的自动构建，在我们还没想好如何构建Agent时可以提供一些灵感。然后介绍如何与GPT4的重要功能代码解释器进行交互。最后，我们再来学习如何定制群聊中的发言逻辑，让自动群聊更加符合预期。



# 材料准备
<div class="alert alert-warning">
    <b>
        注意:
    </b>4.5的代码使用与4.4相同的环境运行，有需要额外的安装会在课件中提到。
</div>

在开始之前我们先提前准备好一些文件：


```python
!mkdir docs
!mkdir sentence-transformers_all-mpnet-base-v2
!curl -o docs/Research.md https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Research.md
!curl -o docs/Spark.md https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md
```

下载[sentence-transformers](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/tree/main)模型，将这个页面上所有文件下载下来，放入src/sentence-transformers_all-mpnet-base-v2文件夹。

下载[xtreme数据](https://huggingface.co/datasets/google/xtreme/tree/main/MLQA.zh.zh)，把其中的validation-00000-of-00001.parquet保存在docs文件夹中，运行处理脚本：



```python
!pip install pyarrow
```

```python
import json
import pyarrow.parquet as pq

# 读取Parquet文件
table = pq.read_table('./docs/validation-00000-of-00001.parquet')

df = table.to_pandas()
data = df.to_json(orient='records', lines=False)

data = json.loads(data)

result = []
for row in data:
    if row['context'] not in result:
        result.append(row['context'])


with open('./docs/xtreme_context.json','w', encoding='utf-8') as out:
    json.dump(result[:87], out,ensure_ascii=False,indent=4)
```

## 一、检索增强的聊天
本章节内容：
- RAG技术介绍
- 使用RetrieveChat进行检索增强代码生成和问答
- RetrieveChat实例
- 定制RetrieveUserProxyAgent 
- RetrieveChat实战：自动群聊

<div class="alert alert-warning">

### AutoGen版本提醒
本章节使用的AutoGen版本为pyautogen==0.6.1，
</div>

&emsp;&emsp;**RAG（Retrieval Augmented Generation,检索增强生成）是一种使用从外部来源获取的事实，来提高生成式 AI 模型准确性和可靠性的技术**。

**LLM的知识更新难题**

&emsp;&emsp;在进入RAG的介绍之前，需要首先理解一个概念，LLM的知识更新是很困难的，主要原因在于：
- LLM的训练数据集是固定的,一旦训练完成就很难再通过继续训练来更新其知识。
- LLM的参数量巨大,随时进行fine-tuning需要消耗大量的资源，并且需要相当长的时间
- LLM的知识是编码在数百亿个参数中的,无法直接查询或编辑其中的知识图谱。

&emsp;&emsp;因此,LLM的知识具有静态、封闭和有限的特点。为了赋予LLM持续学习和获取新知识的能力,RAG应运而生。

**工作原理**

&emsp;&emsp;RAG本质上是通过工程化手段，解决LLM知识更新困难的问题。其核心手段是利用外挂于LLM的知识数据库（通常使用向量数据库）存储未在训练数据集中出现的新数据、领域数据等。通常而言，RAG将知识问答分成三个阶段：索引、知识检索和基于内容的问答。

- 第一阶段是**知识索引**，需要事先将文本数据进行处理,通过词嵌入（Embedding）等向量化技术，将文本映射到低维向量空间，并将向量存储到数据库中，构建起可检索的向量索引。在这个阶段，RAG涉及数据加载器、分割器、向量数据库、提示工程等组件以及LLM本身。
&emsp;&emsp;Embedding是一种表示方法，即把原始输入数据分布地表示成一系列特征的线性组合。比如最典型的例子，我们知道颜色可以使用RGB表示法，这就是一种Embedding表示：

| 颜色   | 局部表示          | 分布式表示              |
| ------ | ---------------- | ----------------------- |
| 琥珀色 | \([1, 0, 0, 0]^T\) | \([1.00, 0.75, 0.00]^T\) |
| 天蓝色 | \([0, 1, 0, 0]^T\) | \([0.00, 0.5, 1.00]^T\)  |
| 中国红 | \([0, 0, 1, 0]^T\) | \([0.67, 0.22, 0.12]^T\) |
| 咖啡色 | \([0, 0, 0, 1]^T\) | \([0.44, 0.31, 0.22]^T\) |

&emsp;&emsp;而具体到NLP中，每一个词都被表示成指定维度（比如300或者768）的向量，每一个维度对应词的一种语义特征。跟颜色的Embedding不同的是，我们知道RGB表示法中三个特征的物理意义，但是在NLP中，我们显然不可能从语言学角度先验地知道每一个维度具体表示哪一种语义特征，也没法知道一个词语对应的特征值具体是多少，所以这就需要通过语言模型训练来得到每个词嵌入（word embedding）。

&emsp;&emsp;得到word embedding之后，可以通过累加、平均、加权等简单的方式得到句子嵌入（sentence embedding），在BERT模型问世后，常见的方式是将CLS向量对应的输出向量作为sentence embedding。

- 第二阶段是**知识检索**，当输入一个问题时，RAG会对知识库进行检索，找到与问题最相关的一批文档。这需要依赖于第一阶段建立的向量索引，根据向量间的相似性进行快速检索。
  
- 第三阶段是**生成答案**，RAG会把输入问题及相应的检索结果文档一起提供给LLM，让LLM充分把这些外部知识融入上下文，并生成相应的答案。RAG控制生成长度,避免生成无关内容。

&emsp;&emsp;这样,LLM就能够充分利用外部知识库的信息,而不需要修改自身的参数。当知识库更新时,新知识也可以通过prompt实时注入到LLM中。这种设计既发挥了LLM强大的语言生成能力,又规避了其知识更新的困境,使之能更智能地回答各类问题,尤其是需要外部知识支持的问题。

&emsp;&emsp;为了理解这一技术，让我们以法庭为例。法官通常根据对法律的一般理解来审理和判决案件。但有些案件需要用到特殊的专业知识，如医疗事故诉讼或劳资纠纷等，因此法官会派法庭书记员去图书馆寻找可以引用的先例和具体案例。与优秀的法官一样，大语言模型（LLM）能够响应人类的各种查询。但为了能够提供引经据典的权威答案，模型需要一个助手来做一些研究。AI 的“法庭书记员”就是一个被称为检索增强生成（RAG）的过程。 

**优点**

RAG的优点主要体现在以下几个方面：

- 可以利用大规模外部知识改进LLM的推理能力和事实性。
- 使用LangChain等框架可以快速实现原型。
- 第一阶段的知识索引可以随时新增数据，延迟非常低，可以忽略不计。因此RAG架构理论上能做到知识的实时更新。
- 可解释性强，RAG可以通过提示工程等技术，使得LLM生成的答案具有更强的可解释性，从而提高了用户对于答案的信任度和满意度。

**缺点**

RAG也有不少缺点：

- 知识检索阶段依赖相似度检索技术，并不是精确检索，因此有可能出现检索到的文档与问题不太相关。
- 在第三阶段生产答案时，由于LLM基于检索出来的知识进行总结，可能缺乏一些基本世界知识，从而导致无法应对用户询问知识库之外的基本问题。
- 向量数据库是一个尚未成熟的技术，缺乏处理大量数据规模的通用方案，因此数据量较大时,速度和性能存在挑战。
- 在推理时需要对用户输入进行预处理和向量化等操作，增加了推理的时间和计算成本。
- 外部知识库的更新和同步，需要投入人力、物力和时间。
- 需要额外的检索组件，增加了架构的复杂度和维护成本。

### 1.1使用RetrieveChat进行检索增强代码生成和问答

&emsp;&emsp;RetrieveChat是AutoGen用于检索增强生成和问答的对话系统。**在这一系列应用中，我们展示如何利用RetrieveChat根据FLAML文档来生成代码和回答问题**。RetrieveChat使用RetrieveAssistantAgent(检索增强助手)和RetrieveUserProxyAgent(检索增强用户代理)，类似于前面AssistantAgent和UserProxyAgent的用法（例如，带有代码生成、执行和调试的自动任务解决）。实质上，RetrieveAssistantAgent和RetrieveUserProxyAgent实现了针对RetrieveChat提示的不同自动回复机制。

> [FLAML](https://github.com/microsoft/FLAML)是一个轻量级的Python库，用于高效自动化机器学习和人工智能操作。它可以根据大型语言模型、机器学习模型等自动化工作流程，并优化其性能。

&emsp;&emsp;要使用检索增强聊天，需要初始化两个代理，包括检索增强用户代理和检索增强助手。初始化检索增强用户代理需要指定文档集合的路径。随后，**检索增强用户代理可以下载文档，将其分成特定大小的块，计算词嵌入并将它们存储在向量数据库中**。

&emsp;&emsp;一旦启动了聊天，代理将根据以下流程协同进行代码生成或问答：

<div align="center">
 <img src="images/7f5b958dbdd0c6e4f5a70fe1852a9a2ad86e44ad56e2d5bc7a02105679c8b6d0.png" alt="扁平结构"/>
</div>

1. 检索增强用户代理**基于嵌入相似性检索文档块**，并将它们与问题一起发送到检索增强助手。
2. 检索增强助手利用LLM根据提供的问题和上下文生成代码或文本作为答案。如果LLM无法生成满意的响应，它会被指示回复“更新上下文”给检索增强用户代理。
3. 如果响应包含代码块，检索增强用户代理执行代码并将输出作为反馈发送。如果没有代码块或更新上下文的指令，则终止对话。否则，它会更新上下文并将问题与新上下文转发给检索增强助手。请注意，如果启用了人类输入请求，个人可以主动发送任何反馈，包括“更新上下文”给检索增强助手。
4. **如果检索增强助手收到“更新上下文”，它会从检索增强用户代理请求下一个最相似的文档块作为新的上下文**。否则，它会根据反馈和聊天历史生成新的代码或文本。如果LLM无法生成答案，它再次回复“更新上下文”。这个过程可以重复多次。如果没有更多文档可用于上下文，对话将终止。


<div class="alert alert-info">
Retrievechat官方文档：https://microsoft.github.io/autogen/0.2/docs/notebooks/agentchat_RetrieveChat
</div>

&emsp;&emsp;检索的文档储存在向量数据库中，以支持高效的存储和检索功能。可以存储在向量数据库实例中的接受的文件格式包括：

```python
from autogen.retrieve_utils import TEXT_FORMATS

print("`docs_path` 可接受的文档类型:")
print(TEXT_FORMATS)
```

我们首先初始化RetrieveAssistantAgent和RetrieveUserProxyAgent。system_message设置为“您是一个有用的助手”，用于RetrieveAssistantAgent。详细的请求则在用户消息中给出。generate_init_prompt将说明和检索增强生成任务组合成一个初始提示，以发送给LLM助手。

```python
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import chromadb #向量数据库
import os

# 配置魔法地址，对于访问openai的模型是必要的
# import os
# os.environ["http_proxy"]="127.0.0.1:7890"
# os.environ["https_proxy"]="127.0.0.1:7890"

# 加载LLM配置
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST.json")
config_list = config_list_from_json(env_or_file="ds.json")
```

```python
config_list
```

```python
# 创建一个名为"assistant"的RetrieveAssistantAgent实例
assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="您是一个有用的助手。",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list
    }
)
```

```python

from chromadb.utils import embedding_functions

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="src/sentence-transformers_all-mpnet-base-v2"
)
```

```python
# chromadb数据库初始化
# client = chromadb.PersistentClient(path="./tmp/chromadb") 

# 创建一个名为"ragproxyagent"的RetrieveUserProxyAgent实例
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=3,
    # 文档检索配置
    retrieve_config={
        "task": "code",
        "docs_path": [
            "./docs/Research.md", # FLAML文档的引用
            "./docs/Spark.md" # FLAML文档关于如何与spark交互的部分
        ],
        "custom_text_types": ["mdx"],
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "embedding_function": sentence_transformer_ef,  # 使用自定义的embedding_function
        # "embedding_model": "src/sentence-transformers_all-mpnet-base-v2",
        "overwrite": True,  
        "vector_db": "chroma"
    },
    code_execution_config=False,# 这个例子中我们不执行代码
)
```

创建名为“ragproxyagent”的RetrieveUserProxyAgent实例。
**retrieve_config**为文档检索的配置：
- **task**：检索聊天的任务。可能的值包括 "code"、"qa" 和 "default"。不同任务将有不同的系统提示语。默认值是 default，支持代码和问答。
- **client**：chromadb 客户端。如果未提供该键，则将使用默认客户端 chromadb.Client()。如果您想使用其他向量数据库，请扩展此类并覆盖 retrieve_docs 函数。
- **docs_path**是文档目录的路径。它也可以是单个文件的路径或单个文件的URL。默认情况下，它设置为None。
- **collection_name**：数据库collection的名称。在向量数据库中，collection是一组具有相似属性的文档集合，相当于关系型数据库中的表。如果未提供该键，则将使用默认名称 “autogen-docs”
- **model**：用于检索聊天的模型。如果未提供该键，则将使用默认模型 gpt-4
- **task**指示我们正在处理的任务类型。在这个例子中，它是一个“code”任务。
- **chunk_token_size**是用于检索聊天的块token大小。默认情况下，它设置为max_tokens * 0.6，在这里我们将其设置为2000。
- **embedding_model**：用于检索聊天的嵌入模型。如果未提供该键，则将使用默认模型 all-MiniLM-L6-v2。所有可用模型都可以在 https://www.sbert.net/docs/pretrained_models.html 找到。默认模型是一个快速模型。如果您想使用高性能模型，推荐使用 all-mpnet-base-v2。
- **embedding_function**：用于创建向量数据库的嵌入函数。默认为 None，将使用具有给定 embedding_model 的 SentenceTransformer。如果您想使用 OpenAI、Cohere、HuggingFace 或其他嵌入函数，可以在此处传递，遵循 https://docs.trychroma.com/embeddings 中的示例。
- **customized_prompt**：检索聊天的定制提示语。默认为 None。
- **customized_answer_prefix**：检索聊天的定制答案前缀。默认为 ""。如果不为 ""，且答案中没有该定制答案前缀，将触发 Update Context
- **update_context**：如果为 False，将不会触发 Update Context。默认为 True。
- **custom_text_types**是要处理的文件类型列表。默认是前面运行过的autogen.retrieve_utils.TEXT_FORMATS。这仅适用于docs_path目录下的文件。明确包含的文件和URL将被分块，而不管它们的类型如何。在这个例子中，我们将其设置为["mdx"]，以便仅处理Markdown文件。如果在websit/docs中没有包含任何mdx文件，则不会处理任何文件。

通过RetrieveUserProxyAgent发送给assistant的默认提示语，我们也可以进一步理解检索聊天的流程：

> PROMPT_DEFAULT = """ 你是一个增强型检索聊天机器人。你基于自己的知识和用户提供的上下文来回答用户的问题。你应该按照以下步骤来回答问题：
步骤1，根据问题和上下文估计用户的意图。意图可以是代码生成任务或问题回答任务。
步骤2，根据意图进行回复。
如果你无法用当前的上下文回答问题，你应该准确回复 UPDATE CONTEXT。
如果用户的意图是代码生成，你必须遵守以下规则：
规则1：你不能安装任何包，因为所有所需的包都已经安装好了。
规则2：你必须按照以下格式编写你的代码：
\```language
\# your code
\```
如果用户的意图是问题回答，你必须给出尽可能简短的答案。
用户的问题是：{input_question}
上下文是：{input_context}
"""

### 1.2 RetrieveChat实例

#### 应用1：无需人类反馈的检索聊天：基于文档生成代码

&emsp;&emsp;让chatgpt使用最新的框架来写代码不行，因为它的训练语料具有时效性，但是借助检索增强，我们便可以让chatgpt参考文档来写代码。

&emsp;&emsp;问题：如果我想在分类任务中使用FLAML，并希望在30秒内训练模型，我应该使用哪个API？使用Spark并行训练。如果达到时间限制，请强制取消作业。

> [Apache Spark](https://github.com/apache/spark)是用于大规模数据（large-scala data）处理的统一（unified）分析引擎。

```python
# 每次开启一个新对话时都重置agent
assistant.reset()
ragproxyagent.reset()

# 在给定问题的情况下，我们使用ragproxyagent生成一个prompt，作为初始消息发送给助手。
# 助手接收消息并生成回应。回应将被发送回ragproxyagent进行处理。
# 对话会持续进行，直到满足终止条件，在RetrieveChat中，当没有检测到代码块时为终止条件。
# 有人参与时，对话将一直持续，直到用户说“exit”。
code_problem = "如果我想在分类任务中使用FLAML，并希望在30秒内训练模型，我应该使用哪个API？使用Spark并行训练。如果达到时间限制，请强制取消作业。"

# search_string是作为embeddings搜索的额外过滤条件, 在这个例子中，我们只想搜索那些包含"spark"的文档。
ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem=code_problem, search_string="spark")  

```

#### 应用2：无需人类反馈的检索聊天：基于文档回答问题

借助检索增强，我们可以让chatgpt拥有最新的知识，这时再进行提问得到的回复相对会更加正确。

```python
# 重置assistant
assistant.reset()

# 提问
qa_problem = "谁是FLAML的作者?"
ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator,problem=qa_problem)
```

#### 应用3：含人类反馈的检索聊天：基于文档生成代码

这里演示chatgpt生成代码的同时加入我们的反馈，让代码更符合要求。

```python
# 重置assistant
assistant.reset()

# 将`human_input_mode`设置为`ALWAYS`, 让agent在每一步都寻求反馈
ragproxyagent.human_input_mode = "ALWAYS"

# 提问
code_problem = "如何使用FLAML构建一个股票价格的时间序列预测模型？"
ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator,problem=code_problem)
```

这个示例中，助手给出的代码没有满足要求，我们提出将time_budget改为10.

#### 应用4：含人类反馈的检索聊天：基于文档回答问题

这里演示交互地进行问答。

```python
# 重置assistant
assistant.reset()

# 将`human_input_mode`设置为`ALWAYS`, 让agent在每一步都寻求反馈
ragproxyagent.human_input_mode = "ALWAYS"

# 提问
qa_problem = "在FLAML中lgbm_spark有什么作用？"
ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator,problem=qa_problem)
```

#### 应用5：含人类反馈的检索聊天：基于文档回答问题

这里演示当提供的文档内容比较多时，RetrieveChat的功能，即**在检索到的文档中不包含足够信息时自动更新上下文的特性**。

这个例子采用MLQA问答数据集，每个样本包括一段文本，一个问题和对应的答案：

<div align="center">
 <img src="images/xtreme.png" alt="名字"/>
</div> 

我们会选择一些问题并利用RetrieveChat来回答它们。

```python
# 这个例子文字较多，我们换一个便宜的模型
# config_list[0]["model"] = "gpt-3.5-turbo"
# config_list[0]["model"] = "qwen-turbo"

# 使用的数据集原地址：
# corpus_file = "https://datasets-server.huggingface.co/rows?dataset=xtreme&config=MLQA.zh.zh&split=validation&offset=0&length=100"

# 这里仅使用文本部分，去除了问题和答案
corpus_file = "./docs/xtreme_context.json"


ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=10,
    # 文档检索配置
    retrieve_config={
        "task": "qa",
        "docs_path": [
            corpus_file, # FLAML文档的引用
        ],
        "custom_text_types": ["mdx"],
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "embedding_function": sentence_transformer_ef,  # 使用自定义的embedding_function
        # "embedding_model": "src/sentence-transformers_all-mpnet-base-v2",
        "overwrite": True,  # 如果为真，将创建/返回一个用于检索聊天的文档集合。
        "vector_db": "chroma",
        "collection_name": "xtreme",
        "chunk_mode": "one_line"
    },
    code_execution_config=False,# 这个例子中我们不执行代码
)
```

```python
import json

# queries中包含了我们要演示的问题，我们分5个case来演示这5个问题
queries = """{ "text": "玉米哪部分现今用于糠醛的生产？", "answer": ["玉米芯"]}
{ "text": "万维网什么时候开始对所有人免费？", "answer": ["1993年4月30日"]}
{"text": "约翰·坎贝尔在乐队中是什么位置？", "answer": ["贝斯手"]}
{"text": "日产从什么时候开始向全世界出口汽车？", "answer": ["1950年代"]}
{"text": "谁制作了QRpedia？", "answer": ["Roger Bamkin"]}
"""

# 打印一下这5个问题
queries = [json.loads(line) for line in queries.split("\n") if line]
questions = [q["text"] for q in queries]
answers = [q["answer"] for q in queries]
print(questions)
print(answers)
```

```python
for i in range(len(questions)):
    print(f"\n\n>>>>>>>>>>>>  case {i+1}  <<<<<<<<<<<<\n\n")

    # 每开始一轮新的对话都重置assistant
    assistant.reset()
    
    # 提问
    qa_problem = questions[i]
    ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator,problem=qa_problem, n_results=30)
```

&emsp;&emsp;在这个检索过程中，ragproxyagent先给出指令，给出用户问题，然后接在"Context is:"后面的就是检索到的文档。

&emsp;&emsp;assistant 在第一次检索的文档块中如果没有找到答案，就会回复UPDATE CONTEXT，直到在新的文档块中寻找答案。可以看到5个问题全部回答正确。

### 1.3 定制RetrieveUserProxyAgent

&emsp;&emsp;RetrieveUserProxyAgent 可以通过 retrieve_config 进行定制化。有几个参数可以根据不同的使用情况进行配置。在本节中，我们将展示如何定制嵌入函数、文本分割函数和向量数据库。

#### 1.3.1 定制嵌入函数

&emsp;&emsp;默认情况下，将使用Sentence Transformers及其预训练模型来计算嵌入。也许您想要使用OpenAI、Cohere、HuggingFace或其他嵌入函数。

```python
# 这段代码无需演示
from chromadb.utils import embedding_functions

# openai embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key="YOUR_API_KEY",
                model_name="text-embedding-ada-002"
)

# huggingface embedding function
huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key="YOUR_API_KEY",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 创建RetrieveUserProxyAgent
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    retrieve_config={
        "task": "qa",
        "docs_path": "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
        "embedding_function": openai_ef, # 在这里给出embedding_function
    },
)
```

#### 1.3.2 定制文本分割函数

&emsp;&emsp;在我们能够将文档存储到向量数据库之前，需要将文本分割成块。虽然我们在AutoGen中已经实现了一个灵活的文本分割器，但您可能仍然希望使用不同的文本分割器。还有一些现有的文本分割工具可以很好地重复使用。

&emsp;&emsp;例如，您可以使用langchain中的所有文本分割器。

```python
# 这段代码无需演示
from langchain.text_splitter import RecursiveCharacterTextSplitter

# langchain分词器
recur_spliter = RecursiveCharacterTextSplitter(separators=["\n", "\r", "\t"])

# 创建RetrieveUserProxyAgent
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    retrieve_config={
        "task": "qa",
        "docs_path": "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
        "custom_text_split_function": recur_spliter.split_text,# 在这里给出custom_text_split_function
    },
)
```

#### 1.3.3 定制向量数据库

&emsp;&emsp;我们将chromadb作为默认的向量数据库，您也可以通过简单地覆盖 RetrieveUserProxyAgent 的 retrieve_docs 函数来将其替换为任何其他向量数据库。

&emsp;&emsp;例如，您可以像下面这样使用Qdrant：

```python
# 这段代码无需演示
from qdrant_client import QdrantClient

# 初始化Qdrant客户端
client = QdrantClient(url="***", api_key="***")

from litellm import embedding as test_embedding
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from qdrant_client.models import SearchRequest, Filter, FieldCondition, MatchText

# 自定义一个QdrantRetrieveUserProxyAgent类，继承RetrieveUserProxyAgent
class QdrantRetrieveUserProxyAgent(RetrieveUserProxyAgent):
    
    # query_vector_db是我们自定义的函数，不是RetrieveUserProxyAgent的方法
    def query_vector_db(
        self,
        query_texts: List[str],
        n_results: int = 10,
        search_string: str = "",
        **kwargs,
    ) -> Dict[str, Union[List[str], List[List[str]]]]:

        # 计算问题的embedding
        embed_response = test_embedding('text-embedding-ada-002', input=query_texts)

        all_embeddings: List[List[float]] = []

        for item in embed_response['data']:
            all_embeddings.append(item['embedding'])

        # 批量构建query
        search_queries: List[SearchRequest] = []

        for embedding in all_embeddings:
            search_queries.append(# 每个query是一个SearchRequest
                SearchRequest(
                    vector=embedding,
                    filter=Filter(# Filter指定了搜索的条件
                        must=[
                            FieldCondition(
                                key="page_content",
                                match=MatchText(
                                    text=search_string,
                                )
                            )
                        ]
                    ),
                    limit=n_results,
                    with_payload=True,
                )
            )

        # 批量搜索
        search_response = client.search_batch(
            collection_name="{your collection name}",
            requests=search_queries,
        )

        # 返回值的类型为Dict[str, List[List[Any]]], 键需要有"ids" and "documents", 
        # "ids" 是检索到的文档ids，"documents" 是检索到的文档. 其他的键都是可选的.
        # {
        #     "ids": List[string]
        #     "documents": List[List[string]]
        # }
        return {
            "ids": [[scored_point.id for scored_point in batch] for batch in search_response],
            "documents": [[scored_point.payload.get('page_content', '') for scored_point in batch] for batch in search_response],
            "metadatas": [[scored_point.payload.get('metadata', {}) for scored_point in batch] for batch in search_response]
        }

    # 覆写RetrieveUserProxyAgent的retrieve_docs方法
    def retrieve_docs(self, problem: str, n_results: int = 20, search_string: str = "", **kwargs):
        results = self.query_vector_db(
            query_texts=[problem], # 要解决的问题
            n_results=n_results, # 检索出的结果数目
            search_string=search_string, # 检索的关键词
            **kwargs,
        )

        self._results = results


# 创建RetrieveUserProxyAgent
qdrantragagent = QdrantRetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
    retrieve_config={
        "task": "qa",
    }
)

# 提问
qdrantragagent.retrieve_docs("What is Autogen?", n_results=10, search_string="autogen")
```

1. query_vector_db 方法用于根据传入的文本进行嵌入（embedding），然后使用 Qdrant 进行向量检索。
2. retrieve_docs 方法调用了 query_vector_db 方法来获取文档。

关于qdrant向量数据库的使用这里不展开讲了，有兴趣可以自行了解。

### 1.4 RetrieveChat实战：自动群聊

&emsp;&emsp;在群聊中使用 RetrieveUserProxyAgent 几乎与在两个代理对话例子中使用它相同。唯一的区别是您需要使用 RetrieveUserProxyAgent 来初始化对话。在群聊中并不需要 RetrieveAssistantAgent。

&emsp;&emsp;然而，在某些情况下，您可能希望使用另一个代理来初始化对话。这时候就需要从一个函数中调用RetrieveUserProxyAgent。

```python
from autogen import config_list_from_json
# LLM配置
config_list = config_list_from_json(
    "OAI_CONFIG_LIST.json",
    filter_dict={
        "model": ["gpt-4-0613"],
    },
)

config_list = config_list_from_json(
    "qianwen_config.json",
    filter_dict={
        "model": ["qwen-turbo"],
    },
)

# 定义函数调用，这个函数返回检索到的文档字符串并进行简要总结
llm_config = {
    "config_list": config_list,
    "timeout": 60,
    "seed": 42,
}

```

```python
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json, GroupChat, GroupChatManager
import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import chromadb #向量数据库
import os
from typing_extensions import Annotated

```

```python
# 创建boss助手RetrieveUserProxyAgent
boss_aid = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    # 文档检索配置
    retrieve_config={
        "task": "qa",
        "docs_path": [
            "./docs/Research.md", # FLAML文档的引用
            "./docs/Spark.md" # FLAML文档关于如何与spark交互的部分
        ],
        "custom_text_types": ["mdx"],
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "embedding_function": sentence_transformer_ef,  # 使用自定义的embedding_function
        # "embedding_model": "src/sentence-transformers_all-mpnet-base-v2",
        "overwrite": True,  # 如果为真，将创建/返回一个用于检索聊天的文档集合。
        "vector_db": "chroma",
    },
    code_execution_config=False,# 这个例子中我们不执行代码
)



# 让boss_aid检索文档
def retrieve_content(
        message: Annotated[
            str,
            "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
        ],
        n_results: Annotated[int, "number of results"] = 3,
    ) -> str:
    boss_aid.n_results = n_results  # Set the number of results to be retrieved.
    _context = {"problem": message, "n_results": n_results}
    ret_msg = boss_aid.message_generator(boss_aid, None, _context)
    return ret_msg or message
```

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_content",
            "description": "当需要代码生成的时候，进行FLAML相关文档的检索",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "保持原义的简练的消息，可以用于检索文档来生成代码和回答问题。",
                    }
                },
                "required": ["message"],
            }  
        }
    }
]

from openai import OpenAI
import json

client = OpenAI(
        api_key=config_list[0]["api_key"], # 千问的api key
        base_url=config_list[0]["base_url"],
)


def call_with_messages(recipient, messages, sender, config):
    if config["tools"] != []:
        first_response = client.chat.completions.create(
        model="qwen-turbo",
        messages=messages,
        tools=config["tools"]
    )
    else:
        first_response = client.chat.completions.create(
        model="qwen-turbo",
        messages=messages,
    )
    assistant_output = first_response.choices[0].message
    
    print(f"\noutput of First round: {first_response}\n")
    # If the model determines that there is no need to invoke the tool, then print out the assistant"s response directly without making a second call to the model.

    # 如果模型认为不需要调用函数，就直接返回
    if assistant_output.tool_calls is None:
        return True, assistant_output.content
    else:
        # 将返回的函数调用加入消息列表
        messages.append(assistant_output)
        # 提取调用的函数名
        function_name = assistant_output.tool_calls[0].function.name
        # 如果调用
        if function_name == "retrieve_content":
            tool_info = {"name": "retrieve_content", "role": "tool"}
            message= json.loads(assistant_output.tool_calls[0].function.arguments)["message"]
            tool_info["content"] = retrieve_content(message)# 将模型回复放在tool_info["content"]中

    # 输出函数回复
    print(f"Tool info：{tool_info['content']}\n")

    # 将函数输出作为message放入对话历史中
    messages.append({
        "role": "tool",
        "name": function_name,
        "content": tool_info["content"],
        "tool_call_id":assistant_output.tool_calls[0].id
    })

    # 模型总结函数的输出
    second_response = client.chat.completions.create(
        model="qwen-turbo",
        messages=messages,
    )
    
    print(f"Output of second round: {second_response}\n")
    print(f"Final response: {second_response.choices[0].message.content}")
    
    return True, second_response.choices[0].message.content
```

```python
# 创建boss agent
boss = autogen.UserProxyAgent(
    name="Boss",
    human_input_mode="ALWAYS",
    system_message="你是boss， 负责提问和分配任务。",
    code_execution_config=False,
    function_map={
            "retrieve_content": retrieve_content,
        }
)

# 创建一个程序员assistant
coder = AssistantAgent(
    name="Senior_Python_Engineer",
    system_message="当你拿到任务时，首先检索相关文档，然后写代码。任务完成后在新的一行回复 `TERMINATE`作为结尾。",
    llm_config=llm_config
)

# 创建一个代码审阅者assistant
reviewer = AssistantAgent(
    name="Code_Reviewer",
    system_message="你是代码审阅者，负责检查代码工程师写的代码的正确性。完成回答后以`TERMINATE`作为结尾。",
    llm_config=llm_config
)


# 注册回复函数
import autogen

# 为所有agents注册retrieve_content函数.
for agent in [coder, reviewer]:
    agent.register_reply(
        trigger=[autogen.Agent, None], 
        reply_func=call_with_messages,
        # position = 0, # 
        config={"tools": tools} if agent.name == "Senior_Python_Engineer" else {"tools": []}
    )

# groupchat和群聊manager
groupchat = autogen.GroupChat(
    agents=[boss, coder, reviewer], messages=[], max_round=6,speaker_selection_method = 'manual'
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})


# 让boss第一个发言
boss.initiate_chat(
    manager,
    message="怎样使用spark在FLAML中并行训练？查询相关文档并给我一个示例代码。程序员负责写代码，审阅者负责检查正确性。",
)
```

## 二、教会大模型新的技能：TeachableAgent

本章节内容：
- 学习用户信息
- 学习新的事实
- 学习用户偏好
- 学习新的技能

&emsp;&emsp;基于LLM的对话助手可以记住与用户的当前对话，还可以在对话过程中展示在上下文中学习用户教导的能力。但是，一旦对话结束，或者当单个对话变得过长以至于LLM无法有效处理时，助手的记忆和学习就会丢失。在随后的对话中，用户被迫一遍又一遍地重复任何必要的指示。

&emsp;&emsp;TeachableAgent通过将用户教导持久化跨越对话边界存储在作为向量数据库的长期记忆中来解决这些限制。在每次对话结束时，记忆会自动保存到磁盘上，然后在下次对话开始时从磁盘加载。与将所有记忆复制到上下文窗口不同（这将占用宝贵的空间），个别记忆（称为备忘录）会根据需要检索到上下文中。这使用户可以将经常使用的事实和技能仅教授给 TeachableAgent 一次，并在以后的对话中回忆起它们。

&emsp;&emsp;为了对备忘录的存储和检索做出有效的决策，TeachableAgent调用 TextAnalyzerAgent（另一个AutoGen代理）的实例来识别和重塑文本，以便记忆事实、偏好和技能。请注意，这增加了涉及相对较少令牌数量的额外LLM调用，这可能会为用户等待每个响应的时间增加几秒钟。

首先我们创建好agent：

```python
from autogen import UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.capabilities.teachability import Teachability
from autogen import ConversableAgent  #

# 配置文件
filter_dict = {"model": ["deepseek-chat"]}
config_list = config_list_from_json(env_or_file="ds.json", filter_dict=filter_dict)
llm_config={"config_list": config_list, "timeout": 120}

# 使用一个基本Agent
teachable_agent = ConversableAgent(
    name="teachable_agent",  # 名字，可以随便取
    llm_config=llm_config
)

# 实例化一个Teachability对象
teachability = Teachability(
    reset_db=False,  # Use True to force-reset the memo DB, and False to use an existing DB.
    path_to_db_dir="./tmp/interactive/teachability_db"  # Can be any path, but teachable agents in a group chat require unique paths.
)

# 将该能力添加到Agent中
teachability.add_to_agent(teachable_agent)

# 实例化一个UserProxyAgent对象
user = UserProxyAgent("user", human_input_mode="ALWAYS", code_execution_config={"use_docker": False})
```

#### 应用1：学习用户信息

用户可以教导代理有关自己的事实。（请注意，由于LLM的微调，它们可能不愿承认知道个人信息。）

**如果出现OperationalError: attempt to write a readonly database的报错，可能是由于notebook没有创建数据库的权限，尝试先执行Teachable.py先与teachable_agent进行一段对话，以生成相关数据库文件**

```python
# 开始对话
teachable_agent.initiate_chat(user, message="你好，我是一个可以进行学习的助理！有什么可以帮助您的吗？")

```

检验一下assisstant是否记住我了：

```python
# 重置agent
teachable_agent.reset()
user.reset()


# 开始对话
teachable_agent.initiate_chat(user, message="你好，我是一个可以进行学习的助理！有什么可以帮助您的吗？")
```

#### 应用2：学习新的事实

```python
# 重置agent
teachable_agent.reset()
user.reset()

# 开始对话
teachable_agent.initiate_chat(user, message="你好，我是一个可以进行学习的助理！有什么可以帮助您的吗？")
```

检验一下：

```python
# 重置agent
teachable_agent.reset()
user.reset()

# 开始对话
teachable_agent.initiate_chat(user, message="你好，我是一个可以进行学习的助理！有什么可以帮助您的吗？")
```

#### 应用3：学习用户偏好
用户可以教导代理他们偏好事务处理的方式。

```python
# 重置agent
teachable_agent.reset()
user.reset()

# 开始对话
teachable_agent.initiate_chat(user, message="你好，我是一个可以进行学习的助理！有什么可以帮助您的吗？")
```

检验一下：

这里虽然没有像前面那样分点介绍，但确实是包含了user所指定的成立时间，核心业务，企业精神这三个要点。

```python
# 重置agent
teachable_agent.reset()
user.reset()

# 开始对话
teachable_agent.initiate_chat(user, message="你好，我是一个可以进行学习的助理！有什么可以帮助您的吗？")
```

#### 应用4：学习新的技能

用户可以通过教授新技能来拓展可教授代理的能力，以完成具有挑战性的任务。通常最好先描述任务，然后（在同一轮次）提供一些处理任务的提示或建议。

[Sparks of Artificial General Intelligence: Early experiments with GPT-4](https://arxiv.org/abs/2303.12712)论文评估了GPT-4在数学问题上的表现，例如下面的问题，在当时它只能解决32%的问题。我们首先展示了一个失败案例，然后教给代理一个策略，将GPT-4的成功率提高到95%以上。



```python
# 重置agent
teachable_agent.reset()
user.reset()

# 开始对话
teachable_agent.initiate_chat(user, message="你好，我是一个可以进行学习的助理！有什么可以帮助您的吗？")
```

检验一下：

```python
teachable_agent.reset()
teachable_agent.initiate_chat(user, message="你好，我是一个可以进行学习的助理！有什么可以帮助您的吗？")
```

## 三、自动构建多智能体系统：AutoBuild

&emsp;&emsp;在这一小节中，我们介绍了AutoBuild，这是一个能够自动构建用于复杂任务的多Agent系统的流程。具体来说，利用了一个新类叫做AgentBuilder，它将在用户提供构建任务和执行任务描述后，自动完成参与专家Agent的生成以及群组聊天的构建。
<div align="center">
 <img src="images/a65d0860d1b35816f992e01c8d98a6f75879f518cdb7f7c09fd37e3f4eca72a1.png" alt="名字"/>
</div> 


&emsp;&emsp;我们使用配置路径和默认配置创建一个AgentBuilder实例。你也可以指定构建模型和Agent模型，它们分别是用于构建和代理的LLM模型。

```python
from autogen.agentchat.contrib.agent_builder import AgentBuilder

# LLM配置
config_path = 'OAI_CONFIG_LIST.json'

default_llm_config = {
    'temperature': 0
}

# AgentBuilder实例
builder = AgentBuilder(config_file_or_env=config_path, builder_model='gpt-4-0613', agent_model='gpt-4-0613')

# builder = AgentBuilder(config_file_or_env=config_path, builder_model='gpt-4-0613', agent_model='gpt-4-0613')

```

&emsp;&emsp;指定一个构建任务并提供一般描述。建筑任务将帮助构建助手（一个LLM）决定应该建立哪些代理。请注意，你的构建任务应该有任务的一般描述。添加一些具体的例子会更好。

```python
# 任务的描述
building_task = "通过编程找到arxiv上的论文, 并且分析它在某些领域的应用。 例如, 在arxiv上找到最新的关于gpt-4的论文并且找出它在软件领域的潜在应用"
```

&emsp;&emsp;使用 build() 方法让构建管理器（以 builder_model 作为支撑）完成群聊代理的生成。如果你认为需要编码来完成你的任务，你可以使用 coding=True 将用户代理（本地代码解释器）添加到代理列表中，如下所示：

- **agent_list**：生成的AssistantAgent实例列表，如果coding设置为True，那么一个UserProxyAssistant实例会被添加为agent_list的第一个实例
- **agent_configs**：包括agent名字，所用LLM model，system message的agent配置，例如：
```python
[
    {
        "name": "Data_scientist",
        "model": "gpt-4-1106-preview",
        "system_message": "作为一名数据科学家，你的任务是自动检索和分析 arXiv 上的学术论文。运用你的 Python 编程技能，开发脚本来收集必要的信息，比如搜索相关论文、下载它们并处理其内容。运用你的分析和语言能力来解读数据，并推断研究在特定领域内的应用。1.为了整理信息，编写并实现 Python 脚本，用于搜索和与在线资源交互、下载和阅读文件、从文档中提取内容，以及执行其他收集信息的任务。使用打印输出作为后续分析的基础。2.在可能的情况下，使用 Python 脚本以编程方式执行任务，确保结果直接显示。高效和策略性地处理每个任务。有条不紊地进行任务。在没有提供策略的情况下，执行前先概述你的计划。清楚区分通过代码处理的任务和利用你的分析专长的任务。在提供代码时，只包括用于无需用户更改即可运行的 Python 脚本。用户应按原样执行你的脚本，无需进行修改：```python\n# filename: <filename>\n# Python script\nprint(\"Your output\")\n```用户不应执行除运行你提供的脚本之外的任何操作。避免呈现需要用户调整的部分或不完整的脚本。不要要求用户复制粘贴结果；相反，适当时使用 'print' 函数显示输出。监控他们分享的执行结果。如果出现错误，提供经过修正的脚本以便重新运行。如果策略未能解决问题，重新评估你的假设，收集所需的额外细节，并探索替代方法。在成功完成任务并验证结果后，确认已实现所述目标。确保所发现的结果准确有效。在可行的情况下提供支持你结论的证据。满足用户需求并确保所有任务完成后，用“TERMINATE”结束你的协助。"
    },
    ...
]
```

```python
# 构建agnet，返回agent的列表和对应的配置
agent_list, agent_configs = builder.build(building_task, default_llm_config, coding=True)
```

```python
# 让在 build() 中生成的代理在群聊中协作完成任务。
import autogen

def start_task(execution_task: str, agent_list: list, llm_config: dict):
    # LLM配置
    config_list = autogen.config_list_from_json(config_path, filter_dict={"model": ["gpt-4-0613"]})

    # 创建group_chat
    group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=12)
    
    # 创建群聊manager
    manager = autogen.GroupChatManager(
        groupchat=group_chat, llm_config={"config_list": config_list, **llm_config}
    )
    
    # 让agent列表中的第一个人先发言
    agent_list[0].initiate_chat(manager, message=execution_task)

start_task(
    execution_task="编写代码，在arxiv上找到最新的关于gpt的论文，并且找出它在软件领域的潜在应用。",
    agent_list=agent_list,
    llm_config=default_llm_config
)

```

## 四、使用代码解释器的GPTAssistant

本章节内容：
- 数学问题求解
- 使用代码解释器绘图


&emsp;&emsp;gpt4的代码解释器在一个受限执行环境中编写并运行 Python 代码，可以生成图表，处理具有多样化数据和格式的文件。它让你的助手能够迭代地运行代码，解决具有挑战性的编程和数学问题，等等。

&emsp;&emsp;这一小节演示autogen如何利用gpt4的代码解释器。


```python
import autogen

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    file_location=".",
    filter_dict={
        "model": ["gpt-4-0613"],
    },
)
```

#### 应用1：数学问题求解

&emsp;&emsp;通过将 code_interpreter 传递给 tools 参数来启用带有代码解释器的 GPTAssistantAgent，它将编写代码并在一个沙盒中自动执行。代理将从沙盒环境中接收结果，并相应地执行操作。

```python
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen.agentchat import AssistantAgent, UserProxyAgent

# 创建一个使用代码解释器的assistant
gpt_assistant = GPTAssistantAgent(
    name="Coder Assistant",
    llm_config={
        "tools": [
            {
                "type": "code_interpreter"
            }
        ],
        "config_list": config_list,
    },
    instructions="你是解数学问题的专家. 编写代码并运行来解决数学问题. 当任务结束并且没有问题时回复 TERMINATE",
)

# 创建一个user_proxy
user_proxy = UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,  # set to True or image name like "python:3" to use docker
    },
    human_input_mode="ALWAYS",
)

# 开始对话
user_proxy.initiate_chat(gpt_assistant, message="假设 $725x + 727y = 1500$ 且 $729x+ 731y = 1508$, 求$x - y$。")
```

#### 应用2：使用代码解释器绘图

&emsp;&emsp;代码解释器可以输出文件，比如生成图像图表。在这个例子中，我们演示如何绘制图形并下载它。

```python
# 创建一个使用代码解释器的assistant
gpt_assistant = GPTAssistantAgent(
    name="Coder Assistant",
    llm_config={
        "tools": [
            {
                "type": "code_interpreter"
            }
        ],
        "config_list": config_list,
    },
    instructions="你是擅长写python代码来解决问题的专家. 当任务结束并且没有问题时回复 TERMINATE",
)

# 开始对话
user_proxy.initiate_chat(gpt_assistant, message="生成数据并且绘制一条线形图来展示美国人口趋势。展示你是如何用代码解决这个问题的。", clear_history=True)
```

在assistant的回复中给出了文件id，通过以下代码可以进行下载并进行展示：

```python
from PIL import Image
import io
from IPython.display import display

# 通过文件id接收
api_response = gpt_assistant.openai_client.files.with_raw_response.retrieve_content("file-PuV6wWd9KnvpyvQM2dTkXVcV")

# 如果正确接收到文件就展示出来
if api_response.status_code == 200:
    content = api_response.content
    image_data_bytes = io.BytesIO(content)
    image = Image.open(image_data_bytes)
    display(image)
```

## 五、再探群聊

&emsp;&emsp;在上一次课中我们介绍了自动群聊的方式，AutoGen的GroupChat类默认让LLM来决定下一个发言者，但是这种方式的群聊效果受限于LLM的能力，如果LLM没能很好地理解群聊的内容，那么群聊将会变得一片混乱。本节我们介绍如何自己定制发言的逻辑。

### 5.1 覆写select_speaker实现多层次的信息流

<div align="center"><img src="images/select_speaker_setup.png" alt="名字"/></div>

&emsp;&emsp;如图，假设这样一个场景，有一份工作需要两个小组相互进行协助，其中小组A的A2知道x的值，A3知道y的值，工作的目标是让小组B的B2知道x*y等于多少。在这个场景中，我们有如下的限制：
- 小组组长之间可以自由交流。
- 每个小组内部可以自由交流。
  
&emsp;&emsp;当参与一项工作的人员增多，沟通成本会急速增加。而将任务分解为子任务，通过限制小组成员只能与自己小组成员交流，可以减少沟通成本。

```python
import random
from typing import Dict, List

import autogen
from autogen.agentchat.agent import Agent
from autogen.agentchat.assistant_agent import AssistantAgent
from autogen.agentchat.groupchat import GroupChat

# LLM配置
config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    filter_dict={
        "model": ["gpt-4-0613"],
    },
)
config_list_gpt4 = autogen.config_list_from_json(
    "ds.json",
    filter_dict={
        "model": ["deepseek-chat"],
    },
)

llm_config = {"config_list": config_list_gpt4, "cache_seed": 43}
```

&emsp;&emsp;这个例子中，我们使用一个名为CustomGroupChat的类来实现自定义的群聊逻辑控制，这种方式的好处在于可以让发言按照我们所期望的顺序执行：
- 自定义发言者选择逻辑：CustomGroupChat类允许我们定义在groupchat中选择下一个发言者的逻辑。
- 基于内容的发言者选择：这个自定义类允许我们根据上一条消息的内容选择下一个发言者，例如"NEXT: A2"或"TERMINATE"。基础的GroupChat类没有这个能力。
- 基于团队的逻辑：CustomGroupChat启用了基于团队的发言者选择逻辑。它允许下一个发言者从与上一个发言者相同的团队中选择，或者从团队领导者中选择，这是基础的GroupChat类不提供的功能。
- 排除上一个发言者：防止立即重新选择上一个发言者，这使得对话更具动态性。
- 特殊情况处理：CustomGroupChat还可以在其select_speaker方法中直接处理特殊情况，比如终止聊天或切换到'User_proxy'。

下面我们来看看如何实现。

```python
# 我们自定义的CustomGroupChat，继承自GroupChat类
class CustomGroupChat(GroupChat):
    def __init__(self, agents, messages, max_round=10):
        super().__init__(agents, messages, max_round)
        self.previous_speaker = None  # 用于记录上一个发言者

    # 覆写select_speaker
    def select_speaker(self, last_speaker: Agent, selector: AssistantAgent):
        last_message = self.messages[-1] if self.messages else None
        
        # 这一段代码检查最后一句对话是否给出了下一位发言者的建议或者结束对话的建议
        if last_message:
            if "NEXT:" in last_message["content"]: # 检查最后一句对话是否提议了下一位发言者
                suggested_next = last_message["content"].split("NEXT:")[-1].strip()
                print(f"Extracted suggested_next = {suggested_next}")
                try:
                    return self.agent_by_name(suggested_next) # 如果提议了，那么就选这个agent作为下一位发言者
                except ValueError:
                    pass  # 如果建议的agent名称是未定义的，那么就继续下面的流程
            elif "TERMINATE" in last_message["content"]: # 没有提议下一位发言者，检查上一句对话是否建议结束对话
                try:
                    return self.agent_by_name("User_proxy") # 如果建议结束对话，下一次发言交给"User_proxy"
                except ValueError:
                    pass  # 如果'User_proxy'是未定义的（不在群聊成员中）, 继续下面流程

        # 小组组长列表。在我们这个例子中，小组A的组长是A1，小组B的组长是B1
        team_leader_names = [agent.name for agent in self.agents if agent.name.endswith("1")]
        
        # 这一段代码实现的是前面提到的场景限制
        # possible_next_speakers是可能的下一位发言者列表
        if last_speaker.name in team_leader_names:# 小组组长只能和组长以及同组的人交流
            team_letter = last_speaker.name[0]
            possible_next_speakers = [
                agent
                for agent in self.agents
                if (agent.name.startswith(team_letter) or agent.name in team_leader_names)
                and agent != last_speaker
                and agent != self.previous_speaker
            ]
        else: # 小组普通成员只能在小组内部交流
            team_letter = last_speaker.name[0]
            possible_next_speakers = [
                agent
                for agent in self.agents
                if agent.name.startswith(team_letter) and agent != last_speaker and agent != self.previous_speaker
            ]

        # 更新上一位发言者
        self.previous_speaker = last_speaker

        if possible_next_speakers: # 从可能的下一位发言者中随机选择一位
            next_speaker = random.choice(possible_next_speakers)
            return next_speaker
        else:
            return None
```

```python
# 对话结束检测函数
def is_termination_msg(content) -> bool:
    have_content = content.get("content", None) is not None
    if have_content and "TERMINATE" in content["content"]:
        return True
    return False
```

```python
# 初始化我们的5个成员
agents_A = [
    AssistantAgent(
        name="A1",
        system_message="你是小组A的组长A1，你的小组成员包括A2, A3。对于小组B，你只能和组长B1交流。",
        llm_config=llm_config,
    ),
    AssistantAgent(
        name="A2",
        system_message="你是小组A的成员A2，你知道x = 9但是不知道y是多少。与其他人交流来进行协作。",
        llm_config=llm_config,
    ),
    AssistantAgent(
        name="A3",
        system_message="你是小组A的成员A3，你知道y = 5但是不知道x是多少。与其他人交流来进行协作。",
        llm_config=llm_config,
    ),
]

agents_B = [
    AssistantAgent(
        name="B1",
        system_message="你是小组B的组长B1，你的小组成员包括B2. 对于小组A，你只能和组长A1交流。",
        llm_config=llm_config,
    ),
    AssistantAgent(
        name="B2",
        system_message="你是小组B的成员B2。你的任务是找出x和y的值并且计算它们的乘积。一旦你得到了结果, 输出结果并且用单独的一行'TERMINATE'作为结束",
        llm_config=llm_config,
    ),
]
```

```python
# 这里的作用是检测到TERMINATE时结束对话.
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="Terminator admin.",
    code_execution_config=False,
    is_termination_msg=is_termination_msg,
    human_input_mode="NEVER",
)

```

```python
# list_of_agents包括5个agent和一个用于控制结束对话的user_proxy
list_of_agents = agents_A + agents_B
list_of_agents.append(user_proxy)

# 实现group_chat
group_chat = CustomGroupChat(
    agents=list_of_agents, 
    messages=['大家合作，帮助B2完成他的任务。小组A有A1、A2、A3。小组B有B1、B2。\
        只有同一小组的成员可以互相交流。只有小组组长可以互相交流。\
        你必须使用"NEXT: B1"这样的方式来建议与B1交流；你只能建议一个人，不能建议自己或前一个发言者；你也可以选择不建议任何人。\
        每个人说话时只需要说给下一个人听，不用回复上一个人的话。'],
    max_round=30,
)

# 实现群聊manager
manager = autogen.GroupChatManager(groupchat=group_chat, llm_config={
    "config_list": config_list_gpt4,
    "cache_seed": None,
})

# 让B2来开始群聊
agents_B[1].initiate_chat(manager, message="我需要知道x,y的值，大家相互交流")
```

### 5.2 使用图结构建模聊天路径
本章节内容：
- 使用select_speaker实现多层次的信息流
- 使用图结构建模聊天路径

&emsp;&emsp;虽然GroupChat类允许转换到任何代理（有或没有LLM的决策），但某些场景可能需要更多对转换的控制。有向图是控制转换路径的一种可能方式，其中每个节点表示一个代理，每条有向边表示可能的转换路径。使用图结构可以做到精确定义聊天过程中的路径，能在一些复杂的流程中能保证聊天过程的稳定性。

&emsp;&emsp;首先安装依赖：
```sh
pip install networkX~=3.2.1
pip install matplotlib~=3.8.1
```
&emsp;&emsp;让我们来说明具有五个代理的GroupChat的当前转换路径。

```python
import random

import matplotlib.pyplot as plt
import networkx as nx

import autogen
from autogen.agentchat.assistant_agent import AssistantAgent
from autogen.agentchat.groupchat import GroupChat 

```

```python
# 创建一个空的有向图
graph = nx.DiGraph()

# 向图中添加5个节点
for node_id in range(5):
    graph.add_node(node_id, label=str(node_id))


# 为所有节点之间添加一条边
for source_node in range(5):
    for target_node in range(5):
        if source_node != target_node:  # 节点不指向自己
            graph.add_edge(source_node, target_node)

# 绘图
nx.draw(graph, with_labels=True, font_weight="bold")
```

&emsp;&emsp;再看一个例子，仍是5个节点：

```python
# 创建一个空的有向图
graph = nx.DiGraph()

# 创建5个节点
for node_id in range(5):
    graph.add_node(node_id, label=str(node_id))

# 每个节点与节点0之间添加一条双向边
for source_node in range(5):
    target_node = 0
    if source_node != target_node:  # 节点不指向自己
        graph.add_edge(source_node, target_node)
        graph.add_edge(target_node, source_node)

# 绘图
nx.draw(graph, with_labels=True, font_weight="bold")
```

```python
# 创建一个空的有向图
graph = nx.DiGraph()

# 分为A，B，C三组，每组有5个节点
for prefix in ["A", "B", "C"]:
    # 每组添加5个节点
    for i in range(5):
        node_id = f"{prefix}{i}"
        graph.add_node(node_id, label=node_id)

    # 同组节点之间互相连接
    for source_node in range(5):
        source_id = f"{prefix}{source_node}"
        for target_node in range(5):
            target_id = f"{prefix}{target_node}"
            if source_node != target_node:  # 节点不指向自己
                graph.add_edge(source_id, target_id)

# 连接A0和B0
graph.add_edge("A0", "B0")
# 连接B0和C0
graph.add_edge("B0", "C0")

# 绘图
nx.draw(graph, with_labels=True, font_weight="bold")
```

&emsp;&emsp;节点指向自己的情况：

```python
# 创建一个空的有向图
graph = nx.DiGraph()

# 创建两个节点
for source_node in range(2):
    graph.add_node(source_node, label=source_node)

# 2*2个有向边，两个节点之间互相的连接和节点与自己的连接
for source_node in range(2):
    for target_node in range(2):
        graph.add_edge(source_node, target_node)

# 绘图
nx.draw(graph, with_labels=True, font_weight="bold")
```

&emsp;&emsp;接下来，我们使用一个例子来演示怎么让CustomGroupChat按照图定义的方式进行群聊，这个例子包括9名玩家，每位玩家只知道自己有几块巧克力，最终目标是得到9名玩家的巧克力总数。首先，定义我们这个例子中使用的有向图：

```python
# LLM配置
config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    filter_dict={
        "model": ["gpt-4-0613"],
    },
)

# config_list_gpt4 = autogen.config_list_from_json(
#     "ds.json",
#     filter_dict={
#         "model": ["deepseek-chat"],
#     },
# )
llm_config = {"config_list": config_list_gpt4, "cache_seed": 100}

# 创建一个空的有向图
graph = nx.DiGraph()

# 参与游戏的agent列表
agents = []

# 创建图和agent
for prefix in ["A", "B", "C"]:# 分为A，B，C三组
    for i in range(3):# 每组3个节点
        node_id = f"{prefix}{i}" # A0，A1，A2...

        secret_value = random.randint(1, 5)  # 每个节点自己的巧克力数目
        graph.add_node(node_id, label=node_id, secret_value=secret_value) # 添加节点

        # 为每个节点绑定一个AssistantAgent
        agents.append(
            AssistantAgent(
                name=node_id,
                system_message=f"""你是{node_id}。你有 {secret_value} 块巧克力。
玩家名单为 [A0, A1, A2, B0, B1, B2, C0, C1, C2]。
你名字的第一个字符表示你所在的小组，第二个字符表示你是否是小组组长，如果是0则表示是小组组长，否则是小组成员。
约束：小组成员只能在小组内部交流，而小组组长可以与其他小组的组长交流，但不能与其他小组的普通成员交流。
你可以使用'NEXT:'来建议下一个发言者。例如，'NEXT: A1'。
小组组长必须确保他们知道自己小组三名成员巧克力数量的总和，即A0只负责要求A1和A2的巧克力总数，B0只负责要求B1和B2的巧克力总数，C0只负责要求C1和C2的巧克力总数。
你需要输出你的巧克力总数，以便其他人可以检查总计数。
一旦我们从所有九名玩家那里得到了总计数，就将所有三个小组的总计数相加，然后使用TERMINATE终止讨论。""",
                llm_config=llm_config,
            )
        )

    # 同一小组成员之间相互连接
    for source_node in range(3):
        source_id = f"{prefix}{source_node}"
        for target_node in range(3):
            target_id = f"{prefix}{target_node}"
            if source_node != target_node:  # To avoid self-loops
                graph.add_edge(source_id, target_id)

# 将小组组长互相连接
graph.add_edge("A0", "B0")
graph.add_edge("A0", "C0")
graph.add_edge("B0", "A0")
graph.add_edge("B0", "C0")
graph.add_edge("C0", "A0")
graph.add_edge("C0", "B0")


# 指定A0为第一个发言人
graph.nodes["A0"]["first_round_speaker"] = True

# 第一个发言人涂上红色
def get_node_color(node):
    if graph.nodes[node].get("first_round_speaker", False):
        return "red"
    else:
        return "green"


plt.figure(figsize=(12, 10))
pos = nx.spring_layout(graph)  # 获取所有节点的位置

# 绘图
nx.draw(graph, pos, with_labels=True, font_weight="bold", node_color=[get_node_color(node) for node in graph])

# 展示每个成员的巧克力数目
for node, (x, y) in pos.items():
    secret_value = graph.nodes[node]["secret_value"]
    plt.text(x, y + 0.1, s=f"Secret: {secret_value}", horizontalalignment="center",color="red")

plt.show()
```

&emsp;&emsp;接下来我们实现一个使用图结构的CustomGroupChat：

```python
class CustomGroupChat(GroupChat):
    def __init__(self, agents, messages, max_round=10, graph=None):
        super().__init__(agents, messages, max_round)
        self.previous_speaker = None  # 记录上一位发言者
        self.graph = graph  # 这个是我们用来定义发言路径的图
        
    # 覆写select_speaker
    def select_speaker(self, last_speaker, selector):
        self.previous_speaker = last_speaker # 上一位发言者其实就是最后一位发言者，不过select_speaker参数要求是last_speaker

        # 这一段代码与上一节中select_speaker的第一段类似，判断是否有建议的下一位发言者或者结束对话
        last_message = self.messages[-1] if self.messages else None
        suggested_next = None
        if last_message:
            if "NEXT:" in last_message["content"]:
                suggested_next = last_message["content"].split("NEXT: ")[-1].strip()
                suggested_next = suggested_next.replace(".", "").replace(",", "")
                print(f"Suggested next speaker from the last message: {suggested_next}")

            elif "TERMINATE" in last_message["content"]:
                try:
                    return self.agent_by_name("User_proxy")
                except ValueError:
                    print(f"agent_by_name failed suggested_next: {suggested_next}")

        # 打印上一位发言者
        if self.previous_speaker is not None:
            print("Current previous speaker:", self.previous_speaker.name)

        # 如果还未开始群聊，从graph中找到第一位发言者，并作为候选发言者。
        # eligible：符合条件的。我们可以说eligible_speakers是候选发言者名单。
        if self.previous_speaker is None and self.graph is not None:
            eligible_speakers = [
                agent for agent in self.agents if self.graph.nodes[agent.name].get("first_round_speaker", False)
            ]
            print("First round eligible speakers:", [speaker.name for speaker in eligible_speakers])

        # 已经开始群聊后，候选发言者名单设定为上一位发言者的所有后继节点
        elif self.previous_speaker is not None and self.graph is not None:
            eligible_speaker_names = [target for target in self.graph.successors(self.previous_speaker.name)]
            eligible_speakers = [agent for agent in self.agents if agent.name in eligible_speaker_names]
            print("Eligible speakers based on previous speaker:", eligible_speaker_names)

        # 如果没定义graph，那所有人都设定为候选发言人
        else:
            eligible_speakers = self.agents

        # 打印候选发言者名单
        print(
            f"Eligible speakers based on graph and previous speaker {self.previous_speaker.name if self.previous_speaker else 'None'}: {[speaker.name for speaker in eligible_speakers]}"
        )

        next_speaker = None
        # 下面这段代码是从候选发言者名单中选出发言者的逻辑，分3种情况
        if eligible_speakers:
            print("Selecting from eligible speakers:", [speaker.name for speaker in eligible_speakers])
            # 1. 已经有建议的发言者，并且这位也在候选发言者名单中，那么就选他为下一位发言者
            if suggested_next in [speaker.name for speaker in eligible_speakers]:
                print("suggested_next is in eligible_speakers")
                next_speaker = self.agent_by_name(suggested_next)

            else:
                msgs_len = len(self.messages)
                print(f"msgs_len is now {msgs_len}")
                if len(self.messages) > 1:
                    # 2. 让LLM根据对话历史从候选发言者名单中选出下一位发言者
                    print(
                        f"Using LLM to pick from eligible_speakers: {[speaker.name for speaker in eligible_speakers]}"
                    )
                    # selector是内置的用于群聊选择发言者的LLM
                    selector.update_system_message(self.select_speaker_msg(eligible_speakers))
                    _, name = selector.generate_oai_reply(
                        self.messages
                        + [
                            {
                                "role": "system",
                                "content": f"Read the above conversation. Then select the next role from {[agent.name for agent in eligible_speakers]} to play. Only return the role.",
                            }
                        ]
                    )

                    # LLM仅建议一位发言者时才采用LLM的决定
                    mentions = self._mentioned_agents(name, eligible_speakers)
                    if len(mentions) == 1:
                        name = next(iter(mentions))
                        next_speaker = self.agent_by_name(name)

                if next_speaker is None:
                    # 3. 前面方法都不行，那就从候选发言者名单里随机选择
                    next_speaker = random.choice(eligible_speakers)

            print(f"Selected next speaker: {next_speaker.name}")

            # 返回选出的发言者
            return next_speaker
        else:
            # 候选发言者名单为空
            raise ValueError("No eligible speakers found based on the graph constraints.")
```

```python
# 用于判断是否结束对话，与上一节相同
def is_termination_msg(content) -> bool:
    have_content = content.get("content", None) is not None
    if have_content and "TERMINATE" in content["content"]:
        return True
    return False

# 创建user_proxy
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="Terminator admin.",
    code_execution_config=False,
    is_termination_msg=is_termination_msg,
    human_input_mode="NEVER",
)

# 将user_proxy也加入群聊，用于控制结束对话
agents.append(user_proxy)
```

```python
# 创建group_chat
group_chat = CustomGroupChat(agents=agents, messages=[], max_round=20, graph=graph)  

# 创建群聊manager
manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

# 由A0第一个发言
agents[0].initiate_chat(
    manager,
    message="""
        现在我们这个游戏中有9名玩家, 被平均分到了 A, B, C三个小组，每个小组有3个玩家, 包括一位小组组长.
        我们的目标是计算出9名玩家拥有的巧克力总数。
        A0:?, A1:?, A2:?,
        B0:?, B1:?, B2:?,
        C0:?, C1:?, C2:?""",
)
```

# 课程小结

&emsp;&emsp;本节课，我们深入学习了autogen的应用：让agent基于给定的文档来进行协助，这有助于弥补gpt在事实能力上的缺陷。教会大模型新的能力，并让agent具有持久化记忆功能，教会一次后就不用重复再教了。介绍了autogen的自动构建智能体功能。演示了autogen如何与gpt的代码解释器结合。最后，以自定义群聊流程的实例结尾，学完这个实例，同学们可以尝试实现各种各样的多智能体拓扑结构了。

# AI Agent的展望
&emsp;&emsp;AI Agent是人工智能成为基础设施的重要推动力。回顾技术发展史，技术的尽头是成为基础设施，比如电力成为像空气一样不易被人们察觉，但是又必不可少的基础设施，还如云计算等。当然这个要经历以下三个阶段：

&emsp;&emsp;创新与发展阶段–新技术被发明并开始应用；

&emsp;&emsp;普及与应用阶段–随着技术成熟，它开始被广泛应用于各个领域，对社会和经济产生深远影响；

&emsp;&emsp;基础设施阶段–当技术变得普及到几乎无处不在，它就转变成了一种基础设施，已经成为人们日常生活中不可或缺的一部分。

&emsp;&emsp;几乎所有的人都认同，人工智能会成为未来社会的基础设施，而智能体正在促使人工智能基础设施化。Agent软件由于其成本低、能够适应不同的任务和环境，并且能够进行学习和优化，使得它可以被应用于广泛的领域，进而成为各个行业和社会活动的基础支撑。

<div align="center">
 <img src="images/ai应用.png" />
 <br>
 人工智能智能体应用一览图 
</div> 

&emsp;&emsp;Agent下一步可能会朝着两个方向同时迭代。一是与人协助的智能体，通过执行各种任务来协助人类，侧重工具属性；二是拟人化方向的迭代，能够自主决策，具有长期记忆，具备一定的类人格特征，侧重于类人或超人属性。

# AI Agent的挑战

&emsp;&emsp;从技术优化迭代和实现上来看，AI Agent的发展也面临一些瓶颈：

&emsp;&emsp;首先，我们通过OpenAI的GPTs也能看到，LLM的复杂推理能力不够强、延迟过高等问题抑制了Agent应用的真正成熟。这也是接下来业界工程优化和技术科研突破的方向。

&emsp;&emsp;其次，多智能体（Multi-agent）发展仍面临较大困境。多智能体是一个非常复杂的学术研究方向，随着智能体开始普及到大众市场，已经成为重要的技术现实问题。例如，斯坦福的虚拟小镇就包含了25个智能体的多智能体研究。但是小镇框架开源之后，根据开发者的测试一个Agent一天需要消耗20美金价格的token数，因为其需要记忆和行动的思考量非常大。这一价格是比很多人类工作者更高的，需要后续Agent框架和LLM推理侧的双重优化。

&emsp;&emsp;多智能体协同可以组成智能体社会这一最高形态的技术社会系统。在这个社会系统中，智能体能够根据目标和环境变化执行复杂灵活的任务，并与人类及其他智能体进行互动和协作。智能体社会不仅有助于人类探索和拓展物理及虚拟世界，还能增强和扩展人类的能力与体验。而突破多智能体的发展困境，未来人们建立智能体社会（Agent Society）建立的重要前提。

&emsp;&emsp;同时，这些发展趋势预示着AI Agent可能面临诸如安全性与隐私性、伦理与责任、经济和社会就业影响等多方面的挑战。

&emsp;&emsp;（1）安全性和隐私性是智能体的关键特性。这两个因素直接影响AI代理的信任度和控制力。若AI代理出现漏洞、遭受攻击或数据泄露等问题，则可能导致对用户或社会的损害。比如，OpenAI的GPTs在发布后不久，出现了安全漏洞，导致了用户上传的数据泄露。

&emsp;&emsp;（2）伦理和责任是智能体的核心原则。这些原则直接影响智能体的可信度和可控性。若智能体表现出不公平、不透明或不可靠等问题，可能引发用户或社会对技术的排斥。责任归属也是智能体的关键议题，人与智能体协同中的责任归属不清晰或不公正也会带来严重后果。

&emsp;&emsp;（3）经济和社会就业影响。未来工作中的一个重要挑战是人类与智能体之间的竞争。例如，AI自由职业者平台NexusGPT的出现便是对传统自由职业者的冲击。未来的社会工作协同中，也会出现越来越多的智能体，雇主基于效率和效益考虑，可能会尽量减少人力投入。随着智能体技术的成熟，我们必须提前思考这些技术发展对社会和个人职业生涯的长期影响。
<div align="center">
 <img src="images/写作下跌.png" />
 <br>
 以ChatGPT的发布为分水岭，全球自由职业平台上的写作/编辑类从业者的数量和收入都进入了断崖式下跌的轨道
</div>

# Reference 引用
- [autogen文档](https://microsoft.github.io/autogen/docs/Getting-Started)
- [RAG简介](https://zhuanlan.zhihu.com/p/662921096)
- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)
- [AI Agent，为什么是AIGC最后的杀手锏？](https://www.tisi.org/27147)

## 环境配置
<div class="alert alert-info">
    
1. **openai>=1.3.6** <br>
   作用: 与 OpenAI 提供的各种人工智能服务和模型进行交互的 Python 客户端库，也可与仿照 OpenAI api调用格式的其他服务进行交互。
2. **pyarrow>=11.0.0** <br>
   作用: 处理和分析大数据的开源库，提供了高效的内存表示和跨语言的数据交换格式.
3. **chromadb>=0.4.20** <br>
   作用: 开源的嵌入式数据库，特别适用于大型语言模型 (LLM) 和相关应用。它提供了嵌入、向量搜索、文档存储、全文搜索、元数据过滤和多模态处理的功能​ 
4. **langchain>=0.1.1** <br>
   作用: 开发基于大语言模型 (LLM) 应用的框架，通过简化应用生命周期的各个方面，使开发者能够更高效地构建和部署智能应用.
5. **matplotlib>=3.8.2** <br>
   作用: 提供了一种简单而灵活的方法来创建各种静态、动态和交互式的图表和可视化。
6. **networkx>=3.2.1** <br>
   作用: 用于创建、操作和研究复杂网络结构的开源 Python 库，提供了丰富的功能来处理各种类型的图和网络。
7. **io:** <br>
   作用: Python 标准库中的一个模块，用于处理各种类型的输入和输出操作，提供了多种工具来处理文本和二进制数据，支持文件操作、内存缓冲区操作等。
8. **Pillow>=10.1.0** <br>
   作用: 提供了广泛的图像处理功能，可以进行图像的创建、操作和转换。
9. **ipython>=8.18.1** <br>
   作用: 强大的交互式 Python shell，提供了许多增强功能，使 Python 开发更加高效和方便。
