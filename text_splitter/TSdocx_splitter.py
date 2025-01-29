import docx
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

class TSDocTextSplitter:
    def __init__(self):
        self.headings = {}
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1250, chunk_overlap=0)

    def split_text(self, doc: docx.document.Document) -> List[str]:
        headings_content = []
        for para in doc.paragraphs:
            if para.style.name.startswith('Heading'):
                cur_heading_level = int(para.style.name.split()[-1])
                cur_heading = para.text
                self.headings[cur_heading_level] = cur_heading
                
                if cur_heading_level > 1:
                    heading = ''
                    for level in range(1, cur_heading_level):
                        heading += '{last_heading}\n'.format(last_heading=self.headings[level])
                    heading += cur_heading
                    cur_heading = heading

                headings_content.append((cur_heading, []))

            elif headings_content:
                headings_content[-1][1].append(para.text)
        
        split_headdings_content = []
        
        for heading, content in headings_content:
            split_content = self.splitter.split_text('\n'.join(content))
            for c in split_content:
                split_headdings_content.append((heading, c))
        
        
        return self.concatenate_heading_content(split_headdings_content)

    def concatenate_heading_content(self, data):
        return [f"{heading}\n" + ''.join(content) for heading,content in data]