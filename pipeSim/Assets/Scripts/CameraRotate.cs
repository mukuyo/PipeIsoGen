using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraRotate : MonoBehaviour
{
    public Transform target; // 回転する対象のモデル
    public float duration = 5.0f; // 回転にかける時間（秒）
    public int frameRate = 30; // フレームレート
    public float heightIncrement = 2.0f; // 高さの増加量
    public int screenshotWidth = 1920; // スクリーンショットの幅
    public int screenshotHeight = 1080; // スクリーンショットの高さ

    private float currentTime = 0f;
    private int completedCircles = 0; // 完了した円周の数
    private Vector3 initialPosition; // 初期位置を保存
    private Camera cam; // カメラ

    void Start()
    {
        initialPosition = transform.position;
        cam = GetComponent<Camera>();
        StartCoroutine(RotateCamera());
    }

    IEnumerator RotateCamera()
    {
        float anglePerFrame = 360f / (duration * frameRate);
        float totalFrames = duration * frameRate;
        float heightPerFrame = heightIncrement / (3 * totalFrames);

        while (completedCircles < 3) // 3周繰り返す
        {
            currentTime = 0f; // 時間をリセット

            while (currentTime < duration)
            {
                transform.RotateAround(target.position, Vector3.up, anglePerFrame);
                transform.position += new Vector3(0, heightPerFrame, 0); // 高さを徐々に変更
                transform.LookAt(target); // ターゲットを常に注視
                currentTime += 1f / frameRate;

                // スクリーンショットを撮影
                yield return StartCoroutine(CaptureScreenshot($"Assets/Pictures/frame{completedCircles * (int)totalFrames + (int)(currentTime * frameRate)}.jpg"));
                yield return new WaitForSeconds(1f / frameRate);
            }

            completedCircles++;
        }
    }

    IEnumerator CaptureScreenshot(string filename)
    {
        // RenderTextureを作成
        RenderTexture rt = new RenderTexture(screenshotWidth, screenshotHeight, 24);
        cam.targetTexture = rt;
        Texture2D screenShot = new Texture2D(screenshotWidth, screenshotHeight, TextureFormat.RGB24, false);
        
        // カメラから画像をレンダリング
        cam.Render();

        // RenderTextureをアクティブにし、画像を読み取る
        RenderTexture.active = rt;
        screenShot.ReadPixels(new Rect(0, 0, screenshotWidth, screenshotHeight), 0, 0);
        screenShot.Apply();
        
        // RenderTextureとカメラのターゲットテクスチャをリセット
        cam.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);

        // 画像をファイルに保存
        byte[] bytes = screenShot.EncodeToJPG();
        System.IO.File.WriteAllBytes(filename, bytes);

        yield return null;
    }
}
