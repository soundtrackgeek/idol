import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

/** Orbit viewer for one Gaussian splat at a time. */
export class SplatViewer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
  private controls: OrbitControls;
  private mesh: SplatMesh | null = null;
  private pivot = new THREE.Group();
  private home = { position: new THREE.Vector3(0, 0, 3), target: new THREE.Vector3() };
  private loadToken = 0;
  private view: { yaw: number; pitch: number } | null = null;
  private resizeObserver: ResizeObserver;

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    host.appendChild(this.renderer.domElement);

    this.scene.add(new SparkRenderer({ renderer: this.renderer }));
    this.scene.add(this.pivot);
    this.pivot.rotation.set(Math.PI, 0, 0);
    this.setBackground("#15161c");

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.autoRotateSpeed = 1.5;

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(host);
    this.resize();

    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  private resize() {
    const { clientWidth: w, clientHeight: h } = this.host;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  setBackground(color: string) {
    this.scene.background = new THREE.Color(color);
  }

  setAutoRotate(on: boolean) {
    this.controls.autoRotate = on;
  }

  /** Spark renders in the y-down convention of captured 3DGS scenes, while Idol saves y-up objects, so the
   * splat is turned 180 degrees around X by default. "Flip" undoes that, for files in the other convention. */
  setFlip(on: boolean) {
    this.pivot.rotation.set(on ? 0 : Math.PI, 0, 0);
    if (this.mesh) this.frame();
  }

  resetView() {
    this.camera.position.copy(this.home.position);
    this.controls.target.copy(this.home.target);
    this.controls.update();
  }

  clear() {
    this.loadToken++;
    if (this.mesh) {
      this.pivot.remove(this.mesh);
      this.mesh.dispose();
      this.mesh = null;
    }
  }

  get hasSplat() {
    return this.mesh !== null;
  }

  /** Load PLY bytes. `view` is Idol's predicted input camera (radians); the camera starts there so the splat
   * is first seen from the same angle as the source image. Returns the number of splats, or null if a newer
   * load replaced this one. */
  async load(bytes: ArrayBuffer, view: { yaw: number; pitch: number } | null = null): Promise<number | null> {
    this.clear();
    this.view = view;
    const token = this.loadToken;
    const mesh = new SplatMesh({ fileBytes: new Uint8Array(bytes), fileName: "splat.ply" });
    await mesh.initialized;
    if (token !== this.loadToken) {
      mesh.dispose();
      return null;
    }
    this.mesh = mesh;
    this.pivot.add(mesh);
    this.frame();
    return mesh.packedSplats?.numSplats ?? 0;
  }

  /** Direction from the object to the camera. Idol's camera is z-up (yaw = atan2(dx, dy), pitch = asin(dz));
   * save_ply maps (x, y, z) -> (x, -z, y) and the default pivot turn maps (x, y, z) -> (x, -y, -z). */
  private viewDirection() {
    if (!this.view) return new THREE.Vector3(0.35, 0.25, 1).normalize();
    const { yaw, pitch } = this.view;
    const direction = new THREE.Vector3(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch));
    if (this.pivot.rotation.x === 0) direction.set(direction.x, -direction.y, -direction.z); // flipped
    return direction.normalize();
  }

  /** Point the camera at the splat's bounding box, from the input camera's angle when known. */
  private frame() {
    if (!this.mesh) return;
    this.pivot.updateMatrixWorld(true);
    const box = this.mesh.getBoundingBox(true).clone().applyMatrix4(this.mesh.matrixWorld);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length() || 1;
    const distance = size / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))) * 1.05;
    const direction = this.viewDirection();
    this.home.target.copy(center);
    this.home.position.copy(center).addScaledVector(direction, distance);
    this.camera.near = Math.max(distance / 1000, 0.001);
    this.camera.far = distance * 100;
    this.camera.updateProjectionMatrix();
    this.resetView();
  }
}
