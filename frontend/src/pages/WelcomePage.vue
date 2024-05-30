<template>
  <q-page class="flex flex-col content-center w-full h-full min-h-full">
    <div class="flex flex-col my-auto">
      <div class="text-4xl text-center my-5 font-light text-gray-300">
        Welcome to
        <div class="text-6xl my-3 text-white">
          <a class="font-light">RAG</a><a class="font-bold">CHAT</a>
        </div>
        <p class="text-xl max-w-screen-sm">
          Discover a new way to engage with your documents! Upload a PDF or TXT
          file and wait for it to be processed with RAG technology. Ask
          questions and chat with your content effortlessly!
        </p>
      </div>
      <q-btn
        class="m-6 px-12 self-center"
        color="primary"
        text-color="black"
        label="Let's get started!"
        @click="$router.push('/chat')"
      />
    </div>
  </q-page>
</template>
