import { ChatOllama } from 'langchain/chat_models/ollama'
import { PromptTemplate } from 'langchain/prompts'
import { StringOutputParser } from 'langchain/schema/output_parser'
import { RunnableSequence } from 'langchain/schema/runnable'

const promptTemplate =
  PromptTemplate.fromTemplate(`Given the user question below, classify it as either being about \`History\`, \`Art\`, or \`Other\`.
                                     
Do not respond with more than one word.

<question>
{question}
</question>

Classification:`)

const model = new ChatOllama({
  baseUrl: 'http://localhost:11434',
  model: 'llama2'
})

const artQuestion = 'how do I play a guitar?'
const historyQuestion = 'how did world war 2 begin?'
const generalQuestion = 'what is 2 + 2?'

const classificationChain = RunnableSequence.from([
  promptTemplate,
  model,
  new StringOutputParser()
])

async function runClassification (): Promise<void> {
  console.log(await classificationChain.invoke({
    question: artQuestion
  }))
  console.log(await classificationChain.invoke({
    question: historyQuestion
  }))
  console.log(await classificationChain.invoke({
    question: generalQuestion
  }))
}

const historyChain = PromptTemplate.fromTemplate(
  `You are an expert in history.
Always answer questions starting with "As Herodotus told me".
Answer only in one sentence, not a single word more and do not say anything except for the answer.
Respond to the following question:

Question: {question}
Answer:`
).pipe(model)

const artChain = PromptTemplate.fromTemplate(
  `You are an expert in art. \
Always answer questions starting with "As Leonardo Da Vinci told me". \
Answer only in one sentence, not a single word more and do not say anything except for the answer.
Respond to the following question:

Question: {question}
Answer:`
).pipe(model)

const generalChain = PromptTemplate.fromTemplate(
  `You are an expert in everything. \
Always answer questions starting with "As God told me". \
Answer only in one sentence, not a single word more and do not say anything except for the answer.
Respond to the following question:

Question: {question}
Answer:`
).pipe(model)

const route = ({ topic }: { input: string, topic: string }): RunnableSequence => {
  if (topic.toLowerCase().includes('art')) {
    return artChain
  } else if (topic.toLowerCase().includes('history')) {
    return historyChain
  } else {
    return generalChain
  }
}

const fullChain = RunnableSequence.from([
  {
    topic: classificationChain,
    question: (input: { question: string }) => input.question
  },
  route
])

async function runFullChain (): Promise<void> {
  const result1 = await fullChain.invoke({
    question: artQuestion
  })

  console.log(result1)

  const result2 = await fullChain.invoke({
    question: historyQuestion
  })

  console.log(result2)

  const result3 = await fullChain.invoke({
    question: generalQuestion
  })

  console.log(result3)
}

async function main (): Promise<void> {
  await runClassification()

  await runFullChain()
}

void main()
