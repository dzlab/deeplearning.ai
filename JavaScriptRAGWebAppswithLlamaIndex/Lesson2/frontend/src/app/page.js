import styles from "./page.module.css";
import MyComponent from "./my-component.tsx";

export default function Home() {
  return (
    <main className={styles.main}>
      <div className={styles.description}>
        <MyComponent />
      </div>
    </main>
  );
}
