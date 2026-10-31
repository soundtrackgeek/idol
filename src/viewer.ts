import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

type View = { yaw: number; pitch: number } | null;

/** Orbit viewer for one Gaussian splat (or an exported mesh) at a time. */
export class SplatViewer {
  private renderer: THREE.WebGLRenderer;
  private spark: SparkRenderer;
  private scene = new THREE.Scene();
  /** Moved by the orbit controls. */
  private camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
  /** A snapshot of `camera` that the splats were sorted for; frames are only drawn with this one. */
  private renderCamera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
  private controls: OrbitControls;
  private mesh: SplatMesh | null = null;
  private model: THREE.Object3D | null = null;
  private lights = new THREE.Group();
  private pivot = new THREE.Group();
  private home = { position: new THREE.Vector3(0, 0, 3), target: new THREE.Vector3() };
  private loadToken = 0;
  private view: View = null;
  private resizeObserver: ResizeObserver;
  private frameBusy = false;
  /** Frames drawn so far (for measuring the frame rate). */
  framesDrawn = 0;

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    host.appendChild(this.renderer.domElement);

    // Splats must be drawn back to front. Spark sorts them asynchronously (GPU readback + worker), so with
    // automatic updates a fast orbit draws frames with an order from a few frames ago, and the object looks
    // hollow or smeared. We drive updates ourselves and only draw once the sort matches the camera.
    this.spark = new SparkRenderer({ renderer: this.renderer, autoUpdate: false });
    this.scene.add(this.spark);
    this.scene.add(this.pivot);
    this.pivot.rotation.set(Math.PI, 0, 0);
    this.setBackground("#15161c");

    // Lights for mesh previews (splats ignore them).
    this.lights.add(new THREE.HemisphereLight(0xffffff, 0x444455, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(2, 3, 2.5);
    this.lights.add(key);
    const rim = new THREE.DirectionalLight(0xffffff, 0.6);
    rim.position.set(-2, 1, -2.5);
    this.lights.add(rim);
    this.scene.add(this.lights);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.autoRotateSpeed = 1.5;

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(host);
    this.resize();

    this.renderer.setAnimationLoop(() => this.frame());
  }

  /** One animation frame: update the controls, then (if the previous frame is done) sort for this exact
   * camera pose and draw. While a sort runs, the canvas keeps showing the last correct frame. */
  private frame() {
    this.controls.update();
    if (this.frameBusy) return;
    this.frameBusy = true;
    this.renderCamera.copy(this.camera);
    this.renderCamera.updateMatrixWorld();
    this.spark
      .update({ scene: this.scene, camera: this.renderCamera })
      .then(() => {
        this.renderer.render(this.scene, this.renderCamera);
        this.framesDrawn++;
      })
      .catch(() => this.renderer.render(this.scene, this.renderCamera))
      .finally(() => (this.frameBusy = false));
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

  setAutoRotate(on: boolean, speed = 1.5) {
    this.controls.autoRotate = on;
    this.controls.autoRotateSpeed = speed;
  }

  /** Spark renders in the y-down convention of captured 3DGS scenes, while Idol saves y-up objects, so the
   * splat is turned 180 degrees around X by default. "Flip" undoes that, for files in the other convention.
   * Exported meshes are already y-up and are not affected. */
  setFlip(on: boolean) {
    this.pivot.rotation.set(on ? 0 : Math.PI, 0, 0);
    if (this.mesh) this.fit();
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
    if (this.model) {
      this.scene.remove(this.model);
      this.model.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose();
        (Array.isArray(m.material) ? m.material : m.material ? [m.material] : []).forEach((mat) => mat.dispose());
      });
      this.model = null;
    }
  }

  get hasSplat() {
    return this.mesh !== null;
  }

  /** Load splat bytes (.ply, .spz or .splat, from `fileName`). `view` is Idol's predicted input camera
   * (radians); the camera starts there so the splat is first seen from the same angle as the source image.
   * Returns the number of splats, or null if a newer load replaced this one. */
  async load(bytes: ArrayBuffer, view: View = null, fileName = "splat.ply"): Promise<number | null> {
    this.clear();
    this.view = view;
    const token = this.loadToken;
    const mesh = new SplatMesh({ fileBytes: new Uint8Array(bytes), fileName });
    await mesh.initialized;
    if (token !== this.loadToken) {
      mesh.dispose();
      return null;
    }
    this.mesh = mesh;
    this.pivot.add(mesh);
    this.fit();
    return mesh.packedSplats?.numSplats ?? 0;
  }

  /** Load an exported mesh (.glb, y-up, vertex colours). Returns the triangle count. */
  async loadMesh(bytes: ArrayBuffer, view: View = null): Promise<number | null> {
    this.clear();
    this.view = view;
    const token = this.loadToken;
    const gltf = await new GLTFLoader().parseAsync(bytes, "");
    if (token !== this.loadToken) return null;
    let triangles = 0;
    gltf.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      const hasColors = !!m.geometry.getAttribute("color");
      const map = (m.material as THREE.MeshStandardMaterial).map ?? null; // baked texture (game-ready meshes)
      m.material = new THREE.MeshStandardMaterial({ map, vertexColors: hasColors && !map, roughness: 0.75, metalness: 0.0, side: THREE.DoubleSide });
      if (!m.geometry.getAttribute("normal")) m.geometry.computeVertexNormals();
      triangles += (m.geometry.index?.count ?? m.geometry.getAttribute("position").count) / 3;
    });
    this.model = gltf.scene;
    this.scene.add(this.model);
    this.fit();
    return triangles;
  }

  /** Direction from the object to the camera. Idol's camera is z-up (yaw = atan2(dx, dy), pitch = asin(dz)).
   * For splats, save_ply maps (x, y, z) -> (x, -z, y) and the default pivot turn maps (x, y, z) -> (x, -y, -z);
   * exported meshes are already in that upright frame. */
  private viewDirection() {
    if (!this.view) return new THREE.Vector3(0.35, 0.25, 1).normalize();
    const { yaw, pitch } = this.view;
    const direction = new THREE.Vector3(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch));
    if (this.mesh && this.pivot.rotation.x === 0) direction.set(direction.x, -direction.y, -direction.z); // flipped
    return direction.normalize();
  }

  /** Point the camera at the object's bounding box, from the input camera's angle when known. */
  private fit() {
    const object = this.mesh ?? this.model;
    if (!object) return;
    this.pivot.updateMatrixWorld(true);
    const box = this.mesh
      ? this.mesh.getBoundingBox(true).clone().applyMatrix4(this.mesh.matrixWorld)
      : new THREE.Box3().setFromObject(object);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length() || 1;
    const distance = (size / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)))) * 1.05;
    const direction = this.viewDirection();
    this.home.target.copy(center);
    this.home.position.copy(center).addScaledVector(direction, distance);
    this.camera.near = Math.max(distance / 1000, 0.001);
    this.camera.far = distance * 100;
    this.camera.updateProjectionMatrix();
    this.resetView();
  }
}
