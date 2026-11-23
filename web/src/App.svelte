<script>
  // later
</script>

<div class="layout">
  <aside class="sidebar">
    <!-- Selfie Panel -->
  </aside>

  <main>
    <!-- Photo Grid -->
  </main>
</div>

<style>
  .layout{
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 24px;
    align-items: start;
  }

  .sidebar{
    position: sticky; /* selfie panel stays visible while you scroll the grid */
    top: 24px;
  }

  @media (max-width: 700px) {
    .layout{
      grid-template-columns: 1fr;
    }
    .sidebar{
      position: static;
    }
  }
</style>