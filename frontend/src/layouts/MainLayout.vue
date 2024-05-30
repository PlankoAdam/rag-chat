<template>
  <q-layout view="lHh Lpr lFf">
    <q-header elevated>
      <q-toolbar>
        <q-toolbar-title
          class="hover:cursor-pointer select-none text-black flex flex-row no-shadow"
          @click="$router.push('/')"
        >
          <div class="font-light me-0.5">RAG</div>
          <div class="font-bold">CHAT</div>
        </q-toolbar-title>
      </q-toolbar>
    </q-header>

    <q-page-container>
      <router-view />
    </q-page-container>
  </q-layout>
</template>

<script setup></script>
