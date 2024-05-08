import asyncio
import aiohttp
import time
from prompt_generation import load_data, generate_prompts_with_codes
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


# Replace with actual openai key
OPENAI_API_KEY = ""

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}

class ProgressLog:
    def __init__(self, total):
        self.total = total
        self.done = 0
    
    def increment(self):
        self.done = self.done + 1
    
    def __repr__(self):
        return f"Done runs {self.done}/{self.total}."

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5), before_sleep=print, retry_error_callback=lambda _: None)
async def get_completion(content, session, semaphore, progress_log, index):
    async with semaphore:
        print(f"Processing: {progress_log.done + 1}/{progress_log.total}.")
        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json={
            # "model": "gpt-3.5-turbo",
            "model": "gpt-4-turbo-preview",
            "messages": [{"role": "user", "content": content}],
            "temperature": 0
        }) as resp:
            if resp.status != 200:
                error_message = await resp.text()
                print(f"Error from API: {error_message}")
                raise Exception(f"Error with status code {resp.status} from API.")
                # return None
            response_json = await resp.json()
            progress_log.increment()
            print(progress_log)
            return (index, response_json["choices"][0]["message"]["content"])

async def get_completion_list(content_list, max_parallel_calls, timeout=100):
    semaphore = asyncio.Semaphore(value=max_parallel_calls)
    progress_log = ProgressLog(len(content_list))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(timeout)) as session:
        completion_dict = await asyncio.gather(*[get_completion(content, session, semaphore, progress_log, index) for index, content in enumerate(content_list)])
        completion_dict = dict(completion_dict)
        return [completion_dict[i] for i in range(len(content_list))]

async def main():
    sum_prompt = generate_prompts_with_codes("../data/human/human_data.csv", "./prompts.json", "summarization") 
    start_time = time.perf_counter()
    # completion_list = await get_completion_list(["Ping", "Pong"], 100, 1000)

    summarizations = await get_completion_list(sum_prompt, 100, 1000)

    ai_code = await get_completion(summarizations[0], 100, 1000)

    print("Time elapsed: ", time.perf_counter() - start_time, "seconds.")





if __name__ == '__main__':
    asyncio.run(main())