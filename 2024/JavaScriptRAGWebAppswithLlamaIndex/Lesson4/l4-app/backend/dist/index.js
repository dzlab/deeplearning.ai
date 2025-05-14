var __async = (__this, __arguments, generator) => {
  return new Promise((resolve, reject) => {
    var fulfilled = (value) => {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    };
    var rejected = (value) => {
      try {
        step(generator.throw(value));
      } catch (e) {
        reject(e);
      }
    };
    var step = (x) => x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected);
    step((generator = generator.apply(__this, __arguments)).next());
  });
};

// index.ts
import cors from "cors";
import "dotenv/config";
import express2 from "express";

// src/routes/chat.route.ts
import express from "express";

// src/controllers/chat.controller.ts
import { streamToResponse } from "ai";
import { OpenAI } from "llamaindex";

// src/controllers/engine/index.ts
import {
  ContextChatEngine,
  serviceContextFromDefaults,
  storageContextFromDefaults,
  VectorStoreIndex
} from "llamaindex";

// src/controllers/engine/constants.mjs
var STORAGE_CACHE_DIR = "./cache";
var CHUNK_SIZE = 512;
var CHUNK_OVERLAP = 20;

// src/controllers/engine/index.ts
function getDataSource(llm) {
  return __async(this, null, function* () {
    const serviceContext = serviceContextFromDefaults({
      llm,
      chunkSize: CHUNK_SIZE,
      chunkOverlap: CHUNK_OVERLAP
    });
    let storageContext = yield storageContextFromDefaults({
      persistDir: `${STORAGE_CACHE_DIR}`
    });
    const numberOfDocs = Object.keys(
      storageContext.docStore.toDict()
    ).length;
    if (numberOfDocs === 0) {
      throw new Error(
        `StorageContext is empty - call 'npm run generate' to generate the storage first`
      );
    }
    return yield VectorStoreIndex.init({
      storageContext,
      serviceContext
    });
  });
}
function createChatEngine(llm) {
  return __async(this, null, function* () {
    const index = yield getDataSource(llm);
    const retriever = index.asRetriever();
    retriever.similarityTopK = 3;
    return new ContextChatEngine({
      chatModel: llm,
      retriever
    });
  });
}

// src/controllers/llamaindex-stream.ts
import {
  createCallbacksTransformer,
  createStreamDataTransformer,
  experimental_StreamData,
  trimStartOfStreamHelper
} from "ai";
function createParser(res, data, opts) {
  const it = res[Symbol.asyncIterator]();
  const trimStartOfStream = trimStartOfStreamHelper();
  return new ReadableStream({
    start() {
      if (opts == null ? void 0 : opts.image_url) {
        const message = {
          type: "image_url",
          image_url: {
            url: opts.image_url
          }
        };
        data.append(message);
      } else {
        data.append({});
      }
    },
    pull(controller) {
      return __async(this, null, function* () {
        var _a2;
        const { value, done } = yield it.next();
        if (done) {
          controller.close();
          data.append({});
          data.close();
          return;
        }
        const text = trimStartOfStream((_a2 = value.response) != null ? _a2 : "");
        if (text) {
          controller.enqueue(text);
        }
      });
    }
  });
}
function LlamaIndexStream(res, opts) {
  const data = new experimental_StreamData();
  return {
    stream: createParser(res, data, opts == null ? void 0 : opts.parserOptions).pipeThrough(createCallbacksTransformer(opts == null ? void 0 : opts.callbacks)).pipeThrough(createStreamDataTransformer(true)),
    data
  };
}

// src/controllers/chat.controller.ts
var convertMessageContent = (textMessage, imageUrl) => {
  if (!imageUrl)
    return textMessage;
  return [
    {
      type: "text",
      text: textMessage
    },
    {
      type: "image_url",
      image_url: {
        url: imageUrl
      }
    }
  ];
};
var chat = (req, res) => __async(void 0, null, function* () {
  try {
    const { messages, data } = req.body;
    const userMessage = messages.pop();
    if (!messages || !userMessage || userMessage.role !== "user") {
      return res.status(400).json({
        error: "messages are required in the request body and the last message must be from the user"
      });
    }
    const llm = new OpenAI({
      model: process.env.MODEL || "gpt-3.5-turbo"
    });
    const chatEngine = yield createChatEngine(llm);
    const userMessageContent = convertMessageContent(
      userMessage.content,
      data == null ? void 0 : data.imageUrl
    );
    console.log("newMessage", userMessageContent);
    console.log("chathistory", messages);
    const response = yield chatEngine.chat({
      message: userMessageContent,
      chatHistory: messages,
      stream: true
    });
    const { stream, data: streamData } = LlamaIndexStream(response, {
      parserOptions: {
        image_url: data == null ? void 0 : data.imageUrl
      }
    });
    const processedStream = stream.pipeThrough(streamData.stream);
    return streamToResponse(processedStream, res, {
      headers: {
        // response MUST have the `X-Experimental-Stream-Data: 'true'` header
        // so that the client uses the correct parsing logic, see
        // https://sdk.vercel.ai/docs/api-reference/stream-data#on-the-server
        "X-Experimental-Stream-Data": "true",
        "Content-Type": "text/plain; charset=utf-8",
        "Access-Control-Expose-Headers": "X-Experimental-Stream-Data"
      }
    });
  } catch (error) {
    console.error("[LlamaIndex]", error);
    return res.status(500).json({
      error: error.message
    });
  }
});

// src/routes/chat.route.ts
var llmRouter = express.Router();
llmRouter.route("/").post(chat);
var chat_route_default = llmRouter;

// index.ts
var app = express2();
var port = parseInt(process.env.PORT2 || "8000");
var env = process.env["NODE_ENV"];
var isDevelopment = !env || env === "development";
var prodCorsOrigin = process.env["PROD_CORS_ORIGIN"];
app.use(express2.json());
if (isDevelopment) {
  console.warn("Running in development mode - allowing CORS for all origins");
  app.use(cors());
} else if (prodCorsOrigin) {
  console.log(
    `Running in production mode - allowing CORS for domain: ${prodCorsOrigin}`
  );
  const corsOptions = {
    origin: prodCorsOrigin
    // Restrict to production domain
  };
  app.use(cors(corsOptions));
} else {
  console.warn("Production CORS origin not set, defaulting to no CORS.");
}
app.use(express2.text());
app.get("/", (req, res) => {
  res.send("LlamaIndex Express Server");
});
app.use("/api/chat", chat_route_default);
app.listen(port, () => {
  console.log(`\u26A1\uFE0F[server]: Server is running at http://localhost:${port}`);
});
