import * as fs from 'fs';
import * as path from 'path';

export interface LLMResult {
    evalId: string;
    results: {
        prompts: {
            raw: string;
            id: string;
            provider: string;
        }[];
        results: {
            prompt: {
                raw: string;
            };
            promptId: string;
            provider: {
                id: string;
            };
            response: {
                output: string;
            };
        }[];
    };
}

export interface ResponseData {
    prompt: string | {
        role: string;
        content: string;
    }[];
    model: string;
    response: string;
}
export function parseLLMResults(filePath: string) {
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    const jsonData: LLMResult = JSON.parse(fileContent);
    let responseSortByModels: {[llmName: string]: ResponseData[]} = {}
    jsonData.results.results.forEach(result => {
        let promptText: {role: string, content: string}[] | string = result.prompt.raw

        try{
            let parseResult: {role: string, content: string}[] = JSON.parse(result.prompt.raw)
            promptText = parseResult
        } catch{
            ;
        }

        let responseData = {
            prompt: promptText,
            model: result.provider.id,
            response: result.response.output
        }

        if(!(responseData.model in responseSortByModels)){
            responseSortByModels[responseData.model] = []
        }
        responseSortByModels[responseData.model].push(responseData)
    }
);
    
    console.log(responseSortByModels);
    return responseSortByModels;
}

const filePath = path.resolve(__dirname, '../../sample_outputs/sample_output_conversation.json');

console.log("Resolved file path:", filePath);
parseLLMResults(filePath);
