<template>
  <q-page class="flex flex-col content-center w-full h-full min-h-full">
    <div
      v-if="!uploaded"
      class="min-w-fit sm:max-w-xl max-w-2xl w-full flex flex-col min-h-full h-full flex-1 justify-evenly px-8"
    >
      <q-uploader
        class="w-full no-shadow"
        text-color="black"
        url="http://localhost:5000/process"
        label="Upload .pdf/.txt file"
        accept=".pdf, .txt"
        auto-upload
        max-files="1"
        @uploaded="
          (info) => {
            uploaded = true;
            vectorStoreID = JSON.parse(info.xhr.response).vector_store_id;
          }
        "
      />
    </div>

    <div
      v-if="uploaded"
      class="sm:max-w-3xl max-w-2xl w-full flex flex-col min-h-full h-full flex-1"
    >
      <div class="q-pa-md flex flex-col justify-end mb-20">
        <div ref="chatDiv" class="w-full max-h-full">
          <div v-for="(msg, index) in chatHistory" :key="index">
            <div v-if="msg.context" class="italic text-gray-400 my-3 pe-[20%]">
              {{ msg.context }}
            </div>
            <q-chat-message
              :key="index"
              :sent="msg.sent"
              :bg-color="msg.color"
              :text-color="msg.textColor"
            >
              <q-spinner-dots v-if="msg.waiting" />
              <div v-else v-for="(t, index) in msg.text" :key="index">
                {{ t }}
              </div>
            </q-chat-message>
          </div>
        </div>
      </div>
      <div
        class="flex flex-row p-3 fixed bottom-0 chat-input"
        style="width: inherit; max-width: inherit"
      >
        <q-input
          class="flex-1 me-3"
          square
          outlined
          v-model="userText"
          label="Query"
          @keyup.enter="askQuery(userText)"
          :readonly="loading"
        />
        <q-btn
          :loading="loading"
          :disabled="userText.length == 0"
          color="secondary"
          @click="askQuery(userText)"
          label="Send"
        />
      </div>
    </div>
  </q-page>
</template>

<script setup>
import { ref } from "vue";
import axios from "axios";
import { useRouter } from "vue-router";

let router = useRouter();

let uploaded = ref(false);

let loading = ref(false);
let userText = ref("");
let vectorStoreID = ref([]);

let chatHistory = ref([]);
let chatDiv = ref(null);

let msgs = [];

async function askQuery(query) {
  if (!query) return;
  userText.value = "";

  chatHistory.value.push({
    text: [query],
    sent: true,
    waiting: false,
    color: "primary",
    textColor: "black",
  });
  chatHistory.value.push({
    text: [],
    sent: false,
    waiting: true,
    color: "dark",
    textColor: "white",
  });
  scroll();

  loading.value = true;

  let payload = {
    query: query,
    vector_store_id: vectorStoreID.value,
    msgs: msgs,
  };

  axios
    .post("http://localhost:5000/answer", payload)
    .then((response) => {
      msgs.push(query);
      msgs.push(response.data.answer);
      chatHistory.value.pop();
      chatHistory.value.push({
        context: response.data.context,
        text: [response.data.answer],
        sent: false,
        waiting: false,
        color: "dark",
        textColor: "white",
      });
      scroll();
    })
    .catch(() => {
      router.push("/expired");
    })
    .finally(() => {
      loading.value = false;
    });
}

function scroll() {
  chatDiv.value.scrollTop = chatDiv.value.scrollHeight;
}
</script>
