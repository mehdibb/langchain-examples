import { ChatOllama } from 'langchain/chat_models/ollama'
import { PromptTemplate } from 'langchain/prompts'
import { StringOutputParser } from 'langchain/schema/output_parser'
import { RunnableBranch, RunnableSequence } from 'langchain/schema/runnable'

// Example 1: Simple branching

const promptTemplate =
  PromptTemplate.fromTemplate(`Given the user question below, classify it as either being about \`History\`, \`Art\`, or \`Other\`.
                                     
Do not respond with more than one word.

<question>
{question}
</question>

Classification:`)

const artQuestion = 'how to assess a work of art?'
const historyQuestion = 'when did Mongols attack Persia?'
const generalQuestion = 'what is 2 + 2?'

const model = new ChatOllama({
  baseUrl: 'http://localhost:11434',
  model: 'llama2'
})

const classificationChain = RunnableSequence.from([
  promptTemplate,
  model,
  new StringOutputParser()
])

async function runClassification (): Promise<void> {
  await classificationChain.invoke({
    question: artQuestion
  }).then((result) => {
    console.log(`- ${artQuestion} >>>`, result)
  })

  await classificationChain.invoke({
    question: historyQuestion
  }).then((result) => {
    console.log(`- ${historyQuestion} >>>`, result)
  })

  await classificationChain.invoke({
    question: generalQuestion
  }).then((result) => {
    console.log(`- ${generalQuestion} >>>`, result)
  })
}

// Example 2: Branching with multiple chains

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

const branch = RunnableBranch.from([
  [
    (x: { topic: string, question: string }) =>
      x.topic.toLowerCase().includes('art'),
    artChain
  ],
  [
    (x: { topic: string, question: string }) =>
      x.topic.toLowerCase().includes('history'),
    historyChain
  ],
  generalChain
])

const fullChain = RunnableSequence.from([
  {
    topic: classificationChain,
    question: (input: { question: string }) => input.question
  },
  branch
])

async function runBranching (): Promise<void> {
  await fullChain.invoke({
    question: artQuestion
  }).then((result) => {
    console.log(`- ${artQuestion} >>>`, result.content)
  })

  await fullChain.invoke({
    question: historyQuestion
  }).then((result) => {
    console.log(`- ${historyQuestion} >>>`, result.content)
  })

  await fullChain.invoke({
    question: generalQuestion
  }).then((result) => {
    console.log(`- ${generalQuestion} >>>`, result.content)
  })
}

async function main (): Promise<void> {
  await runClassification()

  await runBranching()
}

void main()
