import express from "express";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import fs from "fs";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { QdrantVectorStore } from "@langchain/qdrant";


import dotenv from "dotenv";
import { PDFParse } from "pdf-parse";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
dotenv.config();

const port = process.env.PORT || 3000;

const app = express();

app.use(express.json());

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-3.8-flash",
});



const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001", // 768 dimensions
  taskType: TaskType.RETRIEVAL_DOCUMENT,
  title: "Document title",
});

const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
  url: process.env.QDRANT_URL,
  collectionName: "grocery-store",
});



app.get("/", (req, res) => {
  res.status(200).json({
    message: "Hello , shameem khanam",
  });
});

const upload = async () => {
  const pdfPath = "./knowledge_grocery_store.pdf";
  const buffer = fs.readFileSync(pdfPath);
  const pdfResult = new PDFParse({ data: buffer });
  const textResult = await pdfResult.getText();
  const text = textResult.text;
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200
  });
  const docs = await splitter.createDocuments([text]);
  await vectorStore.addDocuments(docs);

  // console.log(docs);
}
// upload();

app.post("/api/ai", async (req, res) => {
  try {
    const { input } = req.body;

    const docs = await vectorStore.similaritySearch(input, 5);
    // console.log("docs:", docs);
    const context = docs.map((d) => d.pageContent).join("/n");

    const response = await llm.invoke([
      new SystemMessage(`You are a RAG AI assistant.
        STRICT RULES:
        - Answer only from context
        - Do not use outside knowledge
        - If answer not found say:
        " I don't know from uploaded pdf"
        CONTEXT:
        ${context}
        `),
        new HumanMessage(input)
    ]);

    res.status(200).json({
      ai: response.content
    });

  } catch (error) {
    console.error(error);
    return res.status(500).json({
      error: error?.message || "server failed"
    })
  }
});

app.listen(port, () => {
  console.log(`server started on port ${port}`);
});
