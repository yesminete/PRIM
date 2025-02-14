from tqdm import tqdm
from metrics import *
import wandb
from metrics import *


def train(
    model,
    params,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    num_epochs,
    device,
    alpha=(1.0, 1.0, 1.0, 1.0, 1.0),
    scheduler=None,
    chunk_length=128,
    overlap=8,
    project_name="PRIM"
):
    """
    Train and validate the model, logging to wandb.
    """
    # Initialize wandb project
    wandb.init(project=project_name)

    # Log model architecture
    wandb.watch(model, log="all")

    model = model.to(device)

    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 20)

        # Training Phase
        model.train()
        train_loss = 0.0
        train_steps = 0

        metrics = {task: {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0} for task in ["stroke","player","type","role","impact"]}

        for batch in tqdm(train_dataloader, desc="Training"):
            x, y_true = batch
            x = x.to(device)

            # Forward pass
            #padding_masks = (x != 0).any(dim=-1)
            y_pred = model(x)
            y_true = y_true.to(device)

            # Compute loss
            batch_loss = loss_fn(y_pred, y_true, alpha)

            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            max_norm = 1.0
            torch.nn.utils.clip_grad_norm_(params, max_norm)

            optimizer.step()

            train_loss += batch_loss.item()
            train_steps += 1


            # Unpack predictions
            y_pred_stroke, y_pred_player, y_pred_type, y_pred_role, y_pred_impact = y_pred

            # Convert predictions to class outputs
            y_pred_stroke = torch.sigmoid(y_pred_stroke).detach().cpu().numpy()
            y_pred_stroke = select_highest_score_stroke(y_pred_stroke, score_threshold=0.5)
            y_pred_player = torch.argmax(torch.softmax(y_pred_player, dim=-1), dim=-1).detach().cpu().numpy()
            y_pred_type = torch.argmax(torch.softmax(y_pred_type, dim=-1), dim=-1).detach().cpu().numpy()
            y_pred_role = torch.argmax(torch.softmax(y_pred_role, dim=-1), dim=-1).detach().cpu().numpy()
            y_pred_impact = torch.argmax(torch.softmax(y_pred_impact, dim=-1), dim=-1).detach().cpu().numpy()

            # Unpack ground truth
            y_true_stroke = y_true[:, :, 0].unsqueeze(-1).detach().cpu().numpy()
            y_true_player = y_true[:, :, 1].unsqueeze(-1).detach().cpu().numpy()
            y_true_type =  y_true[:, :, 2].unsqueeze(-1).detach().cpu().numpy()
            y_true_role = y_true[:, :, 3].unsqueeze(-1).detach().cpu().numpy()
            y_true_impact = y_true[:, :, 4].unsqueeze(-1).detach().cpu().numpy()
            final_predictions = {
                "stroke": y_pred_stroke,
                "player": y_pred_player,
                "type": y_pred_type,
                "role": y_pred_role,
                "impact": y_pred_impact
            }
            final_labels = {
                "stroke": y_true_stroke,
                "player": y_true_player,
                "type": y_true_type,
                "role": y_true_role,
                "impact": y_true_impact
            }


            for task in metrics.keys():
                if task != "stroke":
                    stroke_mask = (final_labels["stroke"] == 1).squeeze(-1)
                    task_y_true = final_labels[task][stroke_mask]
                    task_y_pred = final_predictions[task][stroke_mask]
                    if len(task_y_true) > 0:
                        task_metrics = compute_metrics(task_y_true, task_y_pred)
                        metrics[task]["precision"] += task_metrics[0]
                        metrics[task]["recall"] += task_metrics[1]
                        metrics[task]["f1"] += task_metrics[2]
                        metrics[task]["accuracy"] += task_metrics[3]
                else:
                    mask = (final_labels["stroke"] != -1)
                    task_y_true = final_labels[task][mask]
                    task_y_pred = final_predictions[task][mask]
                    #print(np.unique(task_y_true))
                    task_metrics = compute_metrics(
                        task_y_true,
                        task_y_pred,
                        task
                    )
                    metrics[task]["precision"] += task_metrics[0]
                    metrics[task]["recall"] += task_metrics[1]
                    metrics[task]["f1"] += task_metrics[2]
                    metrics[task]["accuracy"] += task_metrics[3]


        for task in metrics.keys():
          for metric in metrics[task] :
            metrics[task][metric] /= train_steps

        for task, task_metrics in metrics.items():
          print(f"{task.capitalize()} Train Metrics - Precision: {task_metrics['precision']:.4f}, Recall: {task_metrics['recall']:.4f}, F1: {task_metrics['f1']:.4f}, Accuracy: {task_metrics['accuracy']:.4f}")

        avg_train_loss = train_loss / train_steps
        print(f"Train Loss: {avg_train_loss:.4f}")
        #print(f"log vars:{alpha}")
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **{f"train_{task}_{metric}": value for task, task_metrics in metrics.items() for metric, value in task_metrics.items()}
        })


        if val_dataloader!=None:
          # Validation Phase
          model.eval()
          val_loss = 0.0
          val_steps = 0

          # Initialize containers for chunk predictions and labels
          chunk_predictions = {task: [] for task in ["stroke", "player", "type", "role", "impact"]}
          chunk_labels = {task: [] for task in ["stroke", "player", "type", "role", "impact"]}

          with torch.no_grad():
              for batch in tqdm(val_dataloader, desc="Validation"):
                  x, y_true = batch
                  x = x.to(device)
                  #padding_masks = (x != 0).any(dim=-1)
                  y_true = y_true.to(device)

                  # Forward pass
                  y_pred = model(x)

                  # Compute loss
                  batch_loss = loss_fn(y_pred, y_true, alpha)
                  val_loss += batch_loss.item()
                  val_steps += 1

                  # Extract predictions for each task

                  y_pred_stroke, y_pred_player, y_pred_type, y_pred_role, y_pred_impact = y_pred
                  chunk_predictions["stroke"].append(select_highest_score_stroke(torch.sigmoid(y_pred_stroke).detach().cpu().numpy(), score_threshold=0.5))
                  chunk_predictions["player"].append(torch.argmax(torch.softmax(y_pred_player, dim=-1), dim=-1).cpu().numpy())
                  chunk_predictions["type"].append(torch.argmax(torch.softmax(y_pred_type, dim=-1), dim=-1).cpu().numpy())
                  chunk_predictions["role"].append(torch.argmax(torch.softmax(y_pred_role, dim=-1), dim=-1).cpu().numpy())
                  chunk_predictions["impact"].append(torch.argmax(torch.softmax(y_pred_impact, dim=-1), dim=-1).cpu().numpy())


                  # Extract true labels for each task
                  # True labels for stroke are binary, no need to change
                  y_true_stroke = y_true[:, :, 0].unsqueeze(-1).detach().cpu().numpy()
                  y_true_player = y_true[:, :, 1].unsqueeze(-1).detach().cpu().numpy()
                  y_true_type =  y_true[:, :, 2].unsqueeze(-1).detach().cpu().numpy()
                  y_true_role = y_true[:, :, 3].unsqueeze(-1).detach().cpu().numpy()
                  y_true_impact = y_true[:, :, 4].unsqueeze(-1).detach().cpu().numpy()
                  # Append converted labels to chunk-level containers
                  chunk_labels["stroke"].append(y_true_stroke)
                  chunk_labels["player"].append(y_true_player)
                  chunk_labels["type"].append(y_true_type)
                  chunk_labels["role"].append(y_true_role)
                  chunk_labels["impact"].append(y_true_impact)



          # Aggregate predictions and labels
          final_predictions = {}
          final_labels = {}
          for task in chunk_predictions.keys():
              chunk_predictions[task] = np.concatenate(chunk_predictions[task], axis=0)
              chunk_labels[task] = np.concatenate(chunk_labels[task], axis=0)

              final_predictions[task], final_labels[task] = aggregate_prediction_labels(
                  chunk_predictions[task], chunk_labels[task], chunk_length, overlap
              )
          # Compute metrics on aggregated predictions
          metrics = {task: {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0} for task in final_predictions.keys()}
          for task in metrics.keys():
              if task != "stroke":  # Filter based on stroke predictions
                  stroke_mask = (final_labels["stroke"] == 1)
                  task_y_true = final_labels[task][stroke_mask]
                  task_y_pred = final_predictions[task][stroke_mask]
                  if len(task_y_true) > 0:
                      task_metrics = compute_metrics(task_y_true, task_y_pred)
                      metrics[task]["precision"] += task_metrics[0]
                      metrics[task]["recall"] += task_metrics[1]
                      metrics[task]["f1"] += task_metrics[2]
                      metrics[task]["accuracy"] += task_metrics[3]
              else:
                  mask = (final_labels["stroke"] != -1)
                  task_y_true = final_labels[task][mask]
                  task_y_pred = final_predictions[task][mask]
                  task_metrics = compute_metrics(
                      task_y_true,
                      task_y_pred,
                      task
                  )
                  metrics[task]["precision"] += task_metrics[0]
                  metrics[task]["recall"] += task_metrics[1]
                  metrics[task]["f1"] += task_metrics[2]
                  metrics[task]["accuracy"] += task_metrics[3]

          avg_val_loss = val_loss / val_steps
          print(f"Validation Loss: {avg_val_loss:.4f}")
          for task, task_metrics in metrics.items():
              print(f"{task.capitalize()} validation Metrics - Precision: {task_metrics['precision']:.4f}, Recall: {task_metrics['recall']:.4f}, F1: {task_metrics['f1']:.4f}, Accuracy: {task_metrics['accuracy']:.4f}")

          # Log metrics to wandb
          wandb.log({
              "epoch": epoch,
              "train_loss": avg_train_loss,
              "val_loss": avg_val_loss,
              **{f"validation_{task}_{metric}": value for task, task_metrics in metrics.items() for metric, value in task_metrics.items()}
          })

          # Learning rate scheduler step (optional)
          if scheduler:
              scheduler.step(avg_val_loss)

          # Save the best model
        if epoch%50==0:
            torch.save(model.state_dict(), f"/content/drive/MyDrive/PRIM/models/model_{epoch}.pth")
            print("model saved.")


    print("\nTraining Complete")
    wandb.finish()

