Write-Output "Search and remove Docker images named as 'modules-module_number_service' and 'base_image'"

# Find matching images
$images = docker images --filter "reference=modules-module_*_service" --format "{{.ID}} {{.Repository}}:{{.Tag}}"
$base_image = docker images --filter "reference=demo_base_image" --format "{{.ID}} {{.Repository}}:{{.Tag}}"

# Combine the results of both searches
$all_images = @($images, $base_image) -join "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" } # Remove empty lines

if (-not $all_images) {
    Write-Output "No corresponding images found"
} else {
    Write-Output "Images found:"
    Write-Output $all_images
    
    Write-Output "Removing these images..."
    $all_images -split "`n" | ForEach-Object {
        $image_id = ($_ -split " ")[0]
        if (docker rmi -f $image_id) {
            Write-Output "Successfully removed image: $image_id"
        } else {
            Write-Output "Failed to remove image: $image_id"
        }
    }

    Write-Output "Docker images removed: Ready for a new deployment"
}
