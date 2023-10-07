import { ChatOllama } from 'langchain/chat_models/ollama'
import { BaseOutputParser, type FormatInstructionsOptions } from 'langchain/schema/output_parser'
import { ChatPromptTemplate } from 'langchain/prompts'

import readline from 'node:readline'

/**
 * Parse the output of an LLM call to a comma-separated list.
 */
class CommaSeparatedListOutputParser extends BaseOutputParser<string[]> {
  lc_namespace: string[] = ['langchain', 'comma_separated_list']

  getFormatInstructions (options?: FormatInstructionsOptions | undefined): string {
    throw new Error('Method not implemented.')
  }

  async parse (text: string, callbacks?: any): Promise<string[]> {
    // Split the text by comma and trim each item
    const items = text.split(',').map((item) => item.trim())
    return items
  }
}

const template = `You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more.`

const humanTemplate = '{text}'

/**
 * Chat prompt for generating comma-separated lists. It combines the system
 * template and the human template.
 */
const chatPrompt = ChatPromptTemplate.fromMessages(
  [
    ['system', template],
    ['human', humanTemplate]
  ]
)

const model = new ChatOllama({
  baseUrl: 'http://localhost:11434',
  model: 'orca-mini'
})
const parser = new CommaSeparatedListOutputParser()

const chain = chatPrompt.pipe(model).pipe(parser)

async function main (category: string): Promise<string[]> {
  const result = await chain.invoke({
    text: category
  })

  return result
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
})

rl.question('Please enter a category name: ', (category: string) => {
  void main(category).then((result) => {
    console.log(result)
  })
  rl.close()
})
