<template>
  <q-page class="flex flex-col content-center w-full h-full min-h-full">
    <div class="flex flex-col my-auto">
      <div class="text-4xl text-center my-5 font-light text-gray-300">
        Session has expired.
      </div>
      <q-btn
        class="m-6 px-12 self-center"
        color="primary"
        text-color="black"
        label="Start a new chat"
        @click="$router.push('/chat')"
      />
    </div>
  </q-page>
</template>
